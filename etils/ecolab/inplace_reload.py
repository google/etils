# Copyright 2023 The etils Authors.
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

"""Hot reload."""

from __future__ import annotations

import collections
import contextlib
import enum
import functools
import gc
import inspect
import sys
import types
from typing import Any, Iterator
import weakref

from etils import epy
from etils.ecolab import module_utils
from IPython.extensions import autoreload


class ReloadMode(epy.StrEnum):
  """Invalidate mode.

  When reloading a module, indicate what to do with the old module instance.

  Attributes:
    UPDATE_INPLACE: Update the old module instances with the new reloaded values
    INVALIDATE: Clear the old module so it cannot be reused.
    KEEP_OLD: Do not do anything, so 2 versions of the module coohexist at the
      same time.
  """
  UPDATE_INPLACE = enum.auto()
  INVALIDATE = enum.auto()
  KEEP_OLD = enum.auto()


@functools.cache
def get_reloader() -> _InPlaceReloader:
  """Returns the singleton in-place reloader."""
  return _InPlaceReloader()


class _InPlaceReloader:
  """Global manager which track all reloaded modules."""

  def __init__(self):
    # Previously imported modules
    # When a new module is `reloaded` with `invalidate='replace'`, all previous
    # modules are replaced in-place.
    self._previous_modules: dict[str, list[weakref.ref[types.ModuleType]]] = (
        collections.defaultdict(list)
    )

  @contextlib.contextmanager
  def update_old_modules(
      self,
      *,
      reload: list[str],
      verbose: bool,
      reload_mode: ReloadMode,
  ) -> Iterator[None]:
    """Eventually update old modules."""
    # Track imported modules before they are removed from cache (to update them
    # after reloading)
    self._save_modules(reload=reload)

    # We clear the module cache to trigger a full reload import.
    # This is better than `colab_import.reload_package` as it support
    # reloading modules with complex import dependency tree.
    # The drawback is that module is duplicated between old module instance
    # and re-loaded instance. To make sure the user don't accidentally use
    # previous instances, we're invalidating all previous modules.
    module_utils.clear_cached_modules(
        modules=reload,
        verbose=verbose,
        invalidate=True if reload_mode == ReloadMode.INVALIDATE else False,
    )
    try:
      yield
    finally:
      # After reloading, try to update the reloaded modules
      if reload_mode == ReloadMode.UPDATE_INPLACE:
        _update_old_modules(
            reload=reload,
            previous_modules=self._previous_modules,
            verbose=verbose,
        )

  def _save_modules(self, *, reload: list[str]) -> None:
    """Save all modules."""
    # Save all modules
    for module_name in module_utils.get_module_names(reload):
      module = sys.modules.get(module_name)
      if module is None:
        continue
      self._previous_modules[module_name].append(weakref.ref(module))


def _update_old_modules(
    *,
    reload: list[str],
    previous_modules: dict[str, list[weakref.ref[types.ModuleType]]],
    verbose: bool,
) -> None:
  """Update all old modules."""
  _mock_autoreload()
  for module_name in module_utils.get_module_names(reload):
    new_module = sys.modules[module_name]
    for old_module in previous_modules.get(module_name, []):
      old_module = old_module()  # Resolve weakref
      if old_module is None:  # No ref to update
        if verbose:
          print(f"Skipping {module_name} (dead)")
        continue
      _update_old_module(old_module, new_module, verbose=verbose)


def _update_old_module(
    old_module: types.ModuleType,
    new_module: types.ModuleType,
    *,
    verbose: bool,
) -> None:
  """Mutate the old module version with the new dict.

  This also try to update the class, functions,... from the old module (so
  instances are updated in-place).

  Args:
    old_module: Old module to update
    new_module: New module
    verbose: If `True`, display which objects are updated
  """
  # Mutate individual classes, objects,... so previously created objects uses
  # new code
  for name, old_obj in old_module.__dict__.items():
    if name not in new_module.__dict__:
      # TODO(epot): Could still invalidate the function/class (to prevent
      # missuse) ?
      continue
    # Only update objects part of the module (filter other imported symbols)
    if not _bellong_to_module(old_obj, old_module):
      continue

    new_obj = new_module.__dict__[name]
    if verbose:
      if isinstance(old_obj, type):
        old_obj_cls = old_obj
      else:
        old_obj_cls = type(old_obj)
      print(f"Update {old_obj_cls.__name__} ({new_module.__name__})")
    autoreload.update_generic(old_obj, new_obj)

    # TODO(epot): Also update the `shell.user_ns`.

  # Replace the old dict by the new module content
  old_module.__dict__.clear()
  old_module.__dict__.update(new_module.__dict__)


def _bellong_to_module(obj: Any, module: types.ModuleType) -> bool:
  """Returns `True` if the instance, class, function bellong to module."""
  return hasattr(obj, "__module__") and obj.__module__ == module.__name__


@functools.cache
def _mock_autoreload() -> None:
  """Add `update_instances` to autoreload."""

  autoreload.update_class = _wrap_fn(autoreload.update_class, _new_update_class)
  autoreload.update_generic = _wrap_fn(
      autoreload.update_generic, _new_update_generic
  )

  # Also update tuple
  assert autoreload.UPDATE_RULES[0][1].__name__ == "update_class"
  autoreload.UPDATE_RULES[0] = (
      autoreload.UPDATE_RULES[0][0],
      autoreload.update_class,
  )


def _wrap_fn(old_fn, new_fn):
  # Recover the original function (to support colab reload)
  old_fn = inspect.unwrap(old_fn)

  @functools.wraps(old_fn)
  def decorated(*args, **kwargs):
    return new_fn(old_fn, *args, **kwargs)

  return decorated


def _new_update_class(fn, old, new):
  # update_instances is only present in recent IPython. See:
  # https://github.com/ipython/ipython/blob/main/IPython/extensions/autoreload.py
  fn(old, new)
  update_instances(old, new)


def _new_update_generic(fn, old, new):
  # Stop replacement if the 2 objects are the same
  if old is new:
    return True
  return fn(old, new)


def update_instances(old, new):
  """Backport of `update_instances`."""

  refs = gc.get_referrers(old)

  for ref in refs:
    if type(ref) is old:  # pylint: disable=unidiomatic-typecheck
      object.__setattr__(ref, "__class__", new)
