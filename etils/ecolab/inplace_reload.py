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
import dataclasses
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


@dataclasses.dataclass(frozen=True)
class _ModuleRefs:
  """Reference on the previous module/object instances."""

  modules: list[weakref.ref[types.ModuleType]] = dataclasses.field(
      default_factory=list
  )
  objs: dict[str, list[weakref.ref[Any]]] = dataclasses.field(
      default_factory=lambda: collections.defaultdict(list)
  )

  def save_module(self, module: types.ModuleType) -> None:
    """Save reference on all previous modules/objects."""
    self.modules.append(weakref.ref(module))

    for name, old_obj in module.__dict__.items():
      # Only update objects part of the module (filter other imported symbols)
      if not _belong_to_module(old_obj, module):
        continue

      self.objs[name].append(weakref.ref(old_obj))

  def update_refs_with_new_module(self, new_module: types.ModuleType) -> None:
    """Update all old reference to previous objects."""
    # Update all modules
    for old_module in self.modules:
      old_module = old_module()  # Resolve weakref
      if old_module is None:  # No ref to update
        continue
      _update_old_module(old_module, new_module)

    # Update all objects
    for name, old_obj_refs in self.objs.items():
      if name not in new_module.__dict__:
        # TODO(epot): Could try to invalidate the object
        continue

      new_obj = new_module.__dict__[name]

      for old_obj_ref in old_obj_refs:
        old_obj = old_obj_ref()
        if old_obj is None:
          continue
        # TODO(epot): Support cycles
        _update_generic(old_obj, new_obj)


class _InPlaceReloader:
  """Global manager which track all reloaded modules."""

  def __init__(self):
    # Previously imported modules / objects
    # When a new module is `reloaded` with `UPDATE_INPLACE`, all previous
    # modules are replaced in-place.
    self._previous_modules: dict[str, _ModuleRefs] = collections.defaultdict(
        _ModuleRefs
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
    self._save_objs(reload=reload)

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
        )

  def _save_objs(self, *, reload: list[str]) -> None:
    """Save all modules/objects."""
    # Save all modules
    for module_name in module_utils.get_module_names(reload):
      module = sys.modules.get(module_name)
      if module is None:
        continue

      # Save module with all objects from the module
      self._previous_modules[module_name].save_module(module)


def _update_old_modules(
    *,
    reload: list[str],
    previous_modules: dict[str, _ModuleRefs],
) -> None:
  """Update all old modules."""
  for module_name in module_utils.get_module_names(reload):
    new_module = sys.modules[module_name]
    old_module_refs = previous_modules.get(module_name)
    if old_module_refs is not None:
      old_module_refs.update_refs_with_new_module(new_module)


def _update_old_module(
    old_module: types.ModuleType,
    new_module: types.ModuleType,
) -> None:
  """Mutate the old module version with the new dict.

  This also try to update the class, functions,... from the old module (so
  instances are updated in-place).

  Args:
    old_module: Old module to update
    new_module: New module
  """
  # Replace the old dict by the new module content
  old_module.__dict__.clear()
  old_module.__dict__.update(new_module.__dict__)


def _belong_to_module(obj: Any, module: types.ModuleType) -> bool:
  """Returns `True` if the instance, class, function belong to module."""
  return hasattr(obj, "__module__") and obj.__module__ == module.__name__


def _wrap_fn(old_fn, new_fn):
  # Recover the original function (to support colab reload)
  old_fn = inspect.unwrap(old_fn)

  @functools.wraps(old_fn)
  def decorated(*args, **kwargs):
    return new_fn(old_fn, *args, **kwargs)

  return decorated


def _update_class(old, new):
  """Update the class."""

  for key in list(old.__dict__.keys()):
    old_obj = getattr(old, key)

    try:
      new_obj = getattr(new, key)
    except AttributeError:
      # obsolete attribute: remove it
      try:
        delattr(old, key)
      except (AttributeError, TypeError):
        pass
      continue

    _update_generic(old_obj, new_obj)

    try:
      setattr(old, key, getattr(new, key))
    except (AttributeError, TypeError):
      pass  # skip non-writable attributes

  _update_instances(old, new)


def _update_function(old, new):
  """Upgrade the code object of a function."""
  for name in [
      "__code__",
      "__defaults__",
      "__doc__",
      "__closure__",
      "__globals__",
      "__dict__",
  ]:
    try:
      setattr(old, name, getattr(new, name))
    except (AttributeError, TypeError):
      pass


def _update_property(old, new):
  """Replace get/set/del functions of a property."""
  _update_generic(old.fdel, new.fdel)
  _update_generic(old.fget, new.fget)
  _update_generic(old.fset, new.fset)


def _update_generic(old, new):
  # Stop replacement if the 2 objects are the same
  if old is new:
    return

  for type_, update in _UPDATE_RULES:
    if isinstance(old, type_) and isinstance(new, type_):
      update(old, new)
      return True


_UPDATE_RULES = [
    (type, _update_class),
    (types.FunctionType, _update_function),
    (
        types.MethodType,
        lambda a, b: _update_function(a.__func__, b.__func__),
    ),
    (property, _update_property),
]


def _update_instances(old, new):
  """Backport of `update_instances`."""

  refs = gc.get_referrers(old)

  for ref in refs:
    if type(ref) is old:  # pylint: disable=unidiomatic-typecheck
      object.__setattr__(ref, "__class__", new)
