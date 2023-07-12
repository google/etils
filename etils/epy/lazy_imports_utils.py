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

"""Lazy import utils."""

# Forked from TFDS

# TODO(epot): Could try to unify with
# - etils/ecolab/lazy_utils.py
# - kauldron/konfig/fake_import_utils.py
# - visu3d/utils/py_utils.py
# - tensorflow_datasets/core/utils/lazy_imports_utils.py

from __future__ import annotations

import builtins
import contextlib
import dataclasses
import functools
import importlib
import types
from typing import Any, Iterator


@dataclasses.dataclass
class LazyModule:
  """Module loaded lazily during first call."""

  module_name: str
  _submodules: dict[str, LazyModule] = dataclasses.field(default_factory=dict)

  @functools.cached_property
  def _module(self) -> types.ModuleType:
    return importlib.import_module(self.module_name)

  def __getattr__(self, name: str) -> Any:
    if name in self._submodules:
      # known submodule accessed. Do not trigger import
      return self._submodules[name]
    else:
      return getattr(self._module, name)

  # TODO(epot): Also support __setattr__


def _register_submodule(module: LazyModule, name: str) -> LazyModule:
  child_module = LazyModule(
      module_name=f"{module.module_name}.{name}",
  )
  module._submodules[name] = child_module  # pylint: disable=protected-access
  return child_module


@contextlib.contextmanager
def lazy_imports() -> Iterator[None]:  # pylint: disable=g-doc-args
  """Context Manager which lazy loads packages.

  Their import is not executed immediately, but is postponed to the first
  call of one of their attributes.

  Limitation:

  - You can only lazy load modules (`from x import y` will not work if `y` is a
    constant or a function or a class).

  Usage:

  ```python
  with epy.lazy_imports():
    import tensorflow as tf
  ```

  Yields:
    None
  """
  # Need to mock `__import__` (instead of `sys.meta_path`, as we do not want
  # to modify the `sys.modules` cache in any way)
  original_import = builtins.__import__
  try:
    builtins.__import__ = _lazy_import
    yield
  finally:
    builtins.__import__ = original_import


def _lazy_import(
    name: str,
    globals_=None,
    locals_=None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
):
  """Mock of `builtins.__import__`."""
  del globals_, locals_  # Unused

  if level:
    raise ValueError(f"Relative import statements not supported ({name}).")

  root_name, *parts = name.split(".")
  root = LazyModule(module_name=root_name)

  # Extract inner-most module
  child = root
  for name in parts:
    child = _register_submodule(child, name)

  if fromlist:
    # from x.y.z import a, b

    for fl in fromlist:
      _register_submodule(child, fl)

    return child  # return the inner-most module (`x.y.z`)
  else:
    # import x.y.z
    # import x.y.z as z
    return root  # return the top-level module (`x`)
