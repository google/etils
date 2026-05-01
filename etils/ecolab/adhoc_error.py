# Copyright 2026 The etils Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Better error for adhoc reload."""

import collections
import functools
import sys
import types

import IPython


@functools.cache
def register_better_reload_error() -> None:
  ip = IPython.get_ipython()

  if ip is None:  # In tests
    return

  # What if this conflict with other `ip.set_custom_exc` ?
  # Ideally, should support multiple handlers
  ip.set_custom_exc((NameError, AttributeError), _maybe_better_error)


def _maybe_better_error(self, type_, value, traceback, tb_offset=None):
  """Update the error message."""

  if (
      isinstance(value, (NameError, AttributeError))
      and len(value.args) == 1
      and _is_from_invalidate_module(value)
  ):
    (msg,) = value.args
    value.args = tuple([
        msg
        + "\nYou're trying to use an object created with an old version of a"
        ' module you reloaded. Please re-create the object with the reloaded'
        ' module.'
        + _leaking_modules_hint(value)
    ])
  self.showtraceback(
      (type_, value, traceback),
      tb_offset=tb_offset,
  )


def _leaking_modules_hint(exc: Exception) -> str:
  """Best-effort hint listing modules that hold stale references.

  Scans sys.modules for modules that still reference objects from invalidated
  modules. This can trigger side effects (lazy imports, descriptors, etc.),
  so all exceptions are caught to ensure we never drown out the original error.

  Args:
    exc: The original exception from the invalidated module.

  Returns:
    A hint string to append to the error message, or empty string.
  """
  try:
    invalidated_modules = _get_invalidated_module_names(exc)
    leaking_modules = _find_leaking_modules()

    if leaking_modules:
      summary = _summarize_leaking_modules(leaking_modules)
      if invalidated_modules:
        invalidated_str = ', '.join(sorted(invalidated_modules))
        source = f' from the invalidated module(s) ({invalidated_str})'
      else:
        source = ''
      return (
          '\n\nThe following modules were NOT reloaded but still hold'
          f' stale references{source}:'
          f'\n{summary}'
          '\nConsider adding them to `reload=`.'
      )
  except Exception:  # pylint: disable=broad-except
    pass

  return ''


def _summarize_leaking_modules(modules: list[str]) -> str:
  """Group leaking modules by top-level package for readability."""
  groups = collections.defaultdict(list)
  for mod in modules:
    top_level = mod.split('.')[0]
    groups[top_level].append(mod)

  lines = []
  for top_level in sorted(groups):
    mods = sorted(groups[top_level])
    if len(mods) == 1:
      lines.append(f'  {mods[0]}')
    else:
      lines.append(f'  {top_level}.* ({len(mods)} modules)')

  return '\n'.join(lines)


def _is_from_invalidate_module(exc: Exception) -> bool:
  """Check whether the exception is from an invalidated module."""
  tb = exc.__traceback__
  while tb is not None:
    frame = tb.tb_frame
    if '__etils_invalidated__' in frame.f_globals:
      return True
    tb = tb.tb_next

  return False


def _get_invalidated_module_names(exc: Exception) -> set[str]:
  """Extract the names of invalidated modules from the traceback.

  The `__name__` is preserved during module invalidation (see
  `module_utils._invalidate_module`).

  Args:
    exc: The original exception whose traceback to walk.

  Returns:
    Set of fully-qualified module names found in invalidated frames.
  """
  names = set()
  tb = exc.__traceback__
  while tb is not None:
    frame = tb.tb_frame
    if '__etils_invalidated__' in frame.f_globals:
      module_name = frame.f_globals.get('__name__')
      if module_name:
        names.add(module_name)
    tb = tb.tb_next
  return names


def _find_leaking_modules() -> list[str]:
  """Find modules that hold objects referencing invalidated globals.

  After an adhoc reload with `invalidate=True`, the old module's globals are
  cleared and marked with `__etils_invalidated__`. Other modules that imported
  objects from the reloaded module still hold references to the old objects
  whose methods/metaclass methods have `__globals__` pointing to the cleared
  dict. This function identifies those modules so the user knows what to add
  to `reload=`.

  This scan is only performed after an error has already occurred, so it does
  not affect the performance of successful executions.

  Returns:
    Sorted list of module names that still hold stale references.
  """
  leaking = set()

  for module_name, module in list(sys.modules.items()):
    if module is None:
      continue

    # Skip internal / infrastructure modules to reduce noise
    if module_name.startswith((
        'sys',
        'builtins',
        'importlib',
        'etils.ecolab',
        '_',
    )):
      continue

    if _module_has_stale_refs(module):
      leaking.add(module_name)

  return sorted(leaking)


def _module_has_stale_refs(module: types.ModuleType) -> bool:
  """Check if a module holds objects whose globals have been invalidated."""
  try:
    module_dict = vars(module)
  except TypeError:
    return False

  for attr_name, attr_value in list(module_dict.items()):
    if attr_name.startswith('__'):
      continue

    try:
      if _has_invalidated_globals(attr_value):
        return True
    except Exception:  # pylint: disable=broad-except
      continue

  return False


def _has_invalidated_globals(obj) -> bool:
  """Check if obj or its class/metaclass methods reference invalidated globals.

  Instead of relying on `__module__` name matching (which misses metaclass
  relationships and cross-submodule imports), this directly checks whether
  any function reachable from the object has `__globals__` marked with
  `__etils_invalidated__`.

  Args:
    obj: Any Python object to check.

  Returns:
    True if the object holds a reference to an invalidated globals dict.
  """
  # Direct check: functions/methods
  if _globals_are_invalidated(obj):
    return True

  # For classes: check methods defined on the class and its metaclass
  if isinstance(obj, type):
    for val in vars(obj).values():
      if _globals_are_invalidated(val):
        return True
    # Check metaclass methods (e.g. __getitem__ on a custom metaclass)
    metacls = type(obj)
    if metacls is not type:
      for val in vars(metacls).values():
        if _globals_are_invalidated(val):
          return True

  return False


def _globals_are_invalidated(obj) -> bool:
  """Check if a single callable's __globals__ dict has been invalidated."""
  # Unwrap staticmethod/classmethod to get the underlying function
  if isinstance(obj, (staticmethod, classmethod)):
    obj = obj.__func__
  try:
    func_globals = getattr(obj, '__globals__', None)
  except Exception:  # pylint: disable=broad-except
    return False
  return (
      isinstance(func_globals, dict) and '__etils_invalidated__' in func_globals
  )
