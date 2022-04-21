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
from etils.ecolab.lazy_imports import *
```

To get the list of available modules:

```python
ecolab.lazy_imports.__all__  # List of modules aliases
ecolab.lazy_imports.LAZY_MODULES  # Mapping <module_alias>: <lazy_module info>
```

"""

from __future__ import annotations

# Import as alias to avoid closure issues when updating the global()
import dataclasses as dataclasses_
import importlib as importlib_
import traceback as traceback_
import types as types_
from typing import Any, Optional


# Attributes which will be updated after the module is loaded.
_MODULE_ATTR_NAMES = [
    '__builtins__',
    '__cached__',
    '__doc__',
    '__file__',
    '__loader__',
    '__name__',
    '__package__',
    '__path__',
    '__spec__',
]


@dataclasses_.dataclass(eq=False)
class LazyModuleState:
  """State of the lazy module.

  We store the state in a separate object to:

  1) Reduce the risk of collision
  2) Avoid infinite recursion error when typo on a attribute
  3) `@property`, `epy.cached_property` fail when the class is changed

  """
  module_name: str  # pylint: disable=invalid-name
  host: LazyModule = dataclasses_.field(repr=False)
  _module: Optional[types_.ModuleType] = None
  # Track the trace which trigger the import
  # Helpful to debug.
  # E.g. `Colab` call '.getdoc' on the background, which trigger import.
  trace_repr: Optional[str] = dataclasses_.field(default=None, repr=False)

  @property
  def module(self) -> types_.ModuleType:
    """Returns the module."""
    if not self.module_loaded:  # Load on first call
      # Keep track of attributes which triggered import
      # Used to track ipython internals (e.g. `<module>.get_traits` gets called
      # internally when ipython inspect the object)
      # So writing `<module>.` trigger module loading & auto-completion even if
      # the module was never used before.
      self.trace_repr = ''.join(traceback_.format_stack())

      self._module = _load_module(self.module_name)
      # Update the module.__doc__, module.__file__,...
      self._mutate_host()
    return self._module

  @property
  def module_loaded(self) -> bool:
    return self._module is not None

  def _mutate_host(self) -> None:
    """When the module is first loaded, update `__doc__`, `__file__`,..."""
    assert self.module_loaded
    missing = object()
    for attr_name in _MODULE_ATTR_NAMES:
      attr_value = getattr(self.module, attr_name, missing)
      if attr_value is not missing:
        setattr(self.host, attr_name, attr_value)


# Class name has to be `module` for Colab compatibility (colab hardcodes class
# name instead of checking the instance)
class module(types_.ModuleType):  # pylint: disable=invalid-name
  """Lazy module which auto-loads on first attribute call."""

  def __init__(self, module_name: str):
    # We set `__file__` to None, to avoid `colab_import.reload_package(etils)`
    # to trigger a full reload of all modules here.
    self.__file__ = None

    self._etils_state = LazyModuleState(module_name, host=self)

  def __getattr__(self, name: str) -> Any:
    if not self._etils_state.module_loaded and name in {
        'getdoc',
        '__wrapped__',
    }:
      # IPython dynamically inspect the object when hovering the symbol:
      # This can trigger a slow import which then disable rich annotations:
      # So raising attribute error avoid lazy-loading the module.
      # There might be a more long term fix but this should cover the most
      # common cases.
      raise AttributeError
    return getattr(self._etils_state.module, name)

  def __dir__(self) -> list[str]:  # Used for Colab auto-completion
    return dir(self._etils_state.module)

  def __repr__(self) -> str:
    if not self._etils_state.module_loaded:
      return f'LazyModule({self._etils_state.module_name!r})'
    else:
      module_ = self._etils_state.module
      if hasattr(module_, '__file__'):
        file = module_.__file__
      else:
        file = '(built-in)'
      return f'<lazy_module {module_.__name__!r} from {file!r}>'


# Create alias to avoid confusion
LazyModule = module
del module


# TODO(epot): Rather than hardcoding which modules are adhoc-imported, this
# could be a argument.
def _load_module(module_name: str) -> types_.ModuleType:
  """Load the module, eventually using adhoc-import."""
  import contextlib  # pylint: disable=g-import-not-at-top
  adhoc_cm = contextlib.suppress()

  # First time, load the module
  with adhoc_cm:
    return importlib_.import_module(module_name)


# Modules here are imported from head (missing from the Brain Kernel)
_PACKAGE_RESTRICT = [
    'etils',
    'sunds',
    'visu3d',
    'imageio',
    'mediapy',
]


_STANDARD_MODULE_NAMES = [
    'abc',
    'argparse',
    'ast',
    'asyncio',
    'base64',
    'builtins',
    'collections',
    'colorsys',
    # 'concurrent.futures',
    'contextlib',
    'contextvars',
    'csv',
    'dataclasses',
    'datetime',
    'dis',
    'enum',
    'functools',
    'gc',
    'gzip',
    'html',
    'inspect',
    'io',
    'importlib',
    'itertools',
    'json',
    'math',
    'multiprocessing',
    'os',
    'pathlib',
    'pdb',
    'pickle',
    'pprint',
    'queue',
    'random',
    're',
    'string',
    'subprocess',
    'sys',
    'tarfile',
    'textwrap',
    'threading',
    'time',
    'timeit',
    'traceback',
    'typing',  # Note: With `__future__.annotations`, no need to import Any & co
    'types',
    'uuid',
    'warnings',
    'weakref',
    'zipfile',
]


_MODULE_NAMES = dict(
    # ====== Python standard lib ======
    **{n: n for n in _STANDARD_MODULE_NAMES},
    mock='unittest.mock',
    # ====== Etils ======
    etils='etils',
    array_types='etils.array_types',
    ecolab='etils.ecolab',
    edc='etils.edc',
    enp='etils.enp',
    epath='etils.epath',
    epy='etils.epy',
    etqdm='etils.etqdm',
    etree='etils.etree',  # TODO(epot): etree='etils.etree.jax',
    # ====== Common third party ======
    chex='chex',
    einops='einops',
    flax='flax',
    nn='flax.linen',
    gin='gin',
    imageio='imageio',
    # Even though `import ipywidgets as widgets` is the common alias, widget
    # is likely too ambiguous.
    ipywidgets='ipywidgets',
    jax='jax',
    jnp='jax.numpy',
    matplotlib='matplotlib',
    plt='matplotlib.pyplot',
    media='mediapy',
    np='numpy',
    pd='pandas',
    scipy='scipy',
    sns='seaborn',
    sklearn='sklearn',
    tf='tensorflow',
    tnp='tensorflow.experimental.numpy',
    tfds='tensorflow_datasets',
    tqdm='tqdm',
    tree='tree',
    typing_extensions='typing_extensions',
    plotly='plotly',
    px='plotly.express',
    go='plotly.graph_objects',
    sunds='sunds',
    v3d='visu3d',
)

# Sort the lazy modules per their <module_name>
# Note that this fail with python 3.7, but works with 3.8+
_MODULE_NAMES = dict(sorted(_MODULE_NAMES.items(), key=lambda x: x[1]))

LAZY_MODULES: dict[str, LazyModule] = {
    k: LazyModule(v) for k, v in _MODULE_NAMES.items()
}

globals().update(LAZY_MODULES)

__all__ = sorted(_MODULE_NAMES)  # Sorted per alias
