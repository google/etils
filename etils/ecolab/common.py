# Copyright 2022 The etils Authors.
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

"""Common lazy imports.

Usage:

```python
from etils.ecolab.common import *
```

"""

# TODO(epot): Issues:
# * TF compatibility
# * Add tests
#   Pytest explore `dir(common_test)` to collect the test and is checking
#   for various attributes (e.g., like `_pytestfixturefunction`) which has
#   the side effect of triggering all imports

from __future__ import annotations

import dataclasses
import importlib
import types
from typing import Any

# Track attribute names which trigger the import
# Helpful to debug.
# In `Colab`, colab call '.getdoc' on the background, which trigger import.
_ATTR_NAMES = set()


@dataclasses.dataclass(eq=False)
class LazyModule(types.ModuleType):
  """Lazy module which auto-loads on first attribute call."""
  __module_name: str  # pylint: disable=invalid-name

  def __post_init__(self):
    # We set `__file__` to None, to avoid `colab_import.reload_package(etils)`
    # to trigger a full reload of all modules here.
    self.__file__ = None

  def __getattr__(self, name: str) -> Any:
    _ATTR_NAMES.add(name)
    if '_LazyModule__module_name' not in self.__dict__:
      raise AttributeError
    # First time, load the module
    m = importlib.import_module(self.__module_name)
    # Replace `self` by module (so auto-complete works on colab)
    self.__dict__.clear()
    self.__dict__.update(m.__dict__)
    self.__class__ = type(m)
    # Future call will bypass `__getattr__` entirely (as the class has changed)
    return getattr(m, name)


MODULE_NAMES = dict(
    # ====== Python standard lib ======
    collections='collections',
    dataclasses='dataclasses',
    inspect='inspect',
    os='os',
    pathlib='pathlib',
    sys='sys',
    # With `__future__.annotations`, no need to import Any & co
    typing='typing',
    types='types',
    # ====== Etils ======
    array_types='etils.array_types',
    ecolab='etils.ecolab',
    edc='etils.edc',
    enp='etils.enp',
    epath='etils.epath',
    epy='etils.epy',
    etqdm='etils.etqdm',
    etree='etils.etree',  # TODO(epot): etree='etils.etree.jax',
    # ====== Common third party ======
    gin='gin',
    jax='jax',
    jnp='jax.numpy',
    media='mediapy',
    np='numpy',
    # TODO(epot): TF uses some magic C++ module types with slot not
    # compatible with `LazyModule` (`TFModuleWrapper`)
    # tf='tensorflow',
    # tnp='tensorflow.experimental.numpy',
    tfds='tensorflow_datasets',
    tqdm='tqdm',
    tree='tree',
    px='plotly.express',
    go='plotly.graph_objects',
)

_LAZY_MODULES = {k: LazyModule(v) for k, v in MODULE_NAMES.items()}

globals().update(_LAZY_MODULES)

__all__ = list(MODULE_NAMES)
