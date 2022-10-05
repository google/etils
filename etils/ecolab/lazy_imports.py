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
lazy_imports.__all__  # List of modules aliases
lazy_imports.LAZY_MODULES  # Mapping <module_alias>: <lazy_module info>
```

"""

from __future__ import annotations

# Import as alias to avoid closure issues when updating the global()
import dataclasses as dataclasses_
import importlib as importlib_
import traceback as traceback_
import types as types_
from typing import Any, Optional, Union

from etils import epy as epy_


def __dir__() -> list[str]:  # pylint: disable=invalid-name
  """`lazy_imports` public API.

  Because `globals()` contains hundreds of symbols, we overwrite `dir(module)`
  to avoid poluting the namespace during auto-completion.

  Returns:
    public symbols
  """
  # If modifying this, also update the `lazy_imports/__init__.py``
  return [
      '__all__',
      'LAZY_MODULES',
      'print_current_imports',
      'LazyModule',
      'LazyModuleState',
  ]


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

  Attributes:
    module_name: E.g. `jax.numpy`
    alias: E.g. `jnp`
    host: `LazyModule` attached to the state
    extra_imports: Additional extra imports to trigger (e.g. `concurrent`
      trigger `concurrent.futures` import)
    _module: Cached original imported module
    trace_repr: Track the trace which trigger the import (Helpful to debug)
      E.g. `Colab` call '.getdoc' on the background, which trigger import.
  """

  module_name: str
  alias: str
  host: LazyModule = dataclasses_.field(repr=False)
  extra_imports: list[str] = dataclasses_.field(default_factory=list)
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

      self._module = _load_module(
          self.module_name,
          extra_imports=self.extra_imports,
      )
      # Update the module.__doc__, module.__file__,...
      self._mutate_host()
    return self._module

  @property
  def module_loaded(self) -> bool:
    """Returns `True` if the module is loaded."""
    return self._module is not None

  @property
  def is_std(self) -> bool:
    """Returns `True` if the module is in the standard library."""
    # TODO(epot): Should also contains, `mock`, `concurrent.futures`
    return self.module_name in _STANDARD_MODULE_NAMES

  @property
  def import_statement(self) -> str:
    """Returns the `import xyz` statement."""
    # Possible cases:
    # `import abc.xyz`
    # `import abc.xyz as def`
    # `from abc import xyz`
    # `from abc import xyz as def` (currently, never used)
    if self.module_name == self.alias:
      return f'import {self.module_name}'

    if '.' in self.module_name:
      left_import, right_import = self.module_name.rsplit('.', maxsplit=1)
      if right_import == self.alias:
        return f'from {left_import} import {right_import}'

    # TODO(epot): Also add extra imports ?
    return f'import {self.module_name} as {self.alias}'

  def _mutate_host(self) -> None:
    """When the module is first loaded, update `__doc__`, `__file__`,..."""
    assert self.module_loaded
    missing = object()
    for attr_name in _MODULE_ATTR_NAMES:
      attr_value = getattr(self.module, attr_name, missing)
      if attr_value is not missing:
        object.__setattr__(self.host, attr_name, attr_value)


# Class name has to be `module` for Colab compatibility (colab hardcodes class
# name instead of checking the instance)
class module(types_.ModuleType):  # pylint: disable=invalid-name
  """Lazy module which auto-loads on first attribute call."""

  _etils_state: LazyModuleState

  def __init__(self, module_names: Union[str, list[str]], *, alias: str):
    # We set `__file__` to None, to avoid `colab_import.reload_package(etils)`
    # to trigger a full reload of all modules here.
    object.__setattr__(self, '__file__', None)

    if isinstance(module_names, str):
      module_name = module_names
      extra_imports = []
    else:
      assert isinstance(module_names, list)
      module_name, *extra_imports = module_names

    state = LazyModuleState(
        module_name=module_name,
        alias=alias,
        extra_imports=extra_imports,
        host=self,
    )
    object.__setattr__(self, '_etils_state', state)

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

  def __setattr__(self, name: str, value: Any) -> None:
    # Overwrite the module attribute
    setattr(self._etils_state.module, name, value)

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
def _load_module(
    module_name: str,
    *,
    extra_imports: list[str],
) -> types_.ModuleType:
  """Load the module, eventually using adhoc-import."""
  import contextlib  # pylint: disable=g-import-not-at-top

  adhoc_cm = contextlib.suppress()

  # First time, load the module
  with adhoc_cm:
    for extra_import in extra_imports:
      # Hardcoded hack to not import tqdm.notebook on non-Colab env
      if extra_import == 'tqdm.notebook' and not epy_.is_notebook():
        continue
      importlib_.import_module(extra_import)
    return importlib_.import_module(module_name)


def print_current_imports() -> None:
  """Display the active lazy imports.

  This can be used before publishing a colab. To convert lazy imports
  into explicit imports.

  For convenience, `from etils.ecolab import lazy_imports` is excluded from
  the current imports.

  """
  print(_current_import_statements())


def _current_import_statements() -> str:
  """Returns the lazy import statement string."""
  lines = []

  lazy_modules = [m._etils_state for m in LAZY_MODULES.values()]  # pylint: disable=protected-access
  used_lazy_modules = [
      # For convenience, we do not add the `lazy_imports` import
      m
      for m in lazy_modules
      if m.module_loaded and m.alias != 'lazy_imports'
  ]
  std_modules = [m.import_statement for m in used_lazy_modules if m.is_std]
  non_std_modules = [
      m.import_statement for m in used_lazy_modules if not m.is_std
  ]

  # Import standard python module first, then other modules
  lines.extend(std_modules)
  if std_modules and non_std_modules:
    lines.append('')  # Empty line
  lines.extend(non_std_modules)  # pylint: disable=protected-access
  return '\n'.join(lines)


# Modules here are imported from head (missing from the Brain Kernel)
_PACKAGE_RESTRICT = [
    'dataclass_array',
    'etils',
    'lark',
    'sunds',
    'visu3d',
    'imageio',
    'mediapy',
    'pycolmap',
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
    'copy',
    # 'concurrent.futures',  # Added bellow
    'contextlib',
    'contextvars',
    'csv',
    'dataclasses',
    'datetime',
    'difflib',
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
    'logging',
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
    'stat',
    'string',
    'subprocess',
    'sys',
    'tarfile',
    'textwrap',
    'threading',
    'time',
    'timeit',
    'tomllib',
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
    concurrent=['concurrent', 'concurrent.futures'],
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
    lazy_imports='etils.ecolab.lazy_imports',
    # ====== Common third party ======
    app='absl.app',
    flags='absl.flags',
    beam='apache_beam',
    colabtools='colabtools',
    interactive_forms='colabtools.interactive_forms',
    chex='chex',
    dca='dataclass_array',
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
    lark='lark',
    matplotlib='matplotlib',
    plt='matplotlib.pyplot',
    media='mediapy',
    ml_collections='ml_collections',
    np='numpy',
    pd='pandas',
    pycolmap='pycolmap',
    scipy='scipy',
    sns='seaborn',
    sklearn='sklearn',
    tf='tensorflow',
    tnp='tensorflow.experimental.numpy',
    tfds='tensorflow_datasets',
    tqdm=['tqdm', 'tqdm.auto', 'tqdm.notebook'],
    tree='tree',
    typing_extensions='typing_extensions',
    plotly='plotly',
    px='plotly.express',
    go='plotly.graph_objects',
    sunds='sunds',
    v3d='visu3d',
)

# Note that this fail with python 3.7, but works with 3.8+

LAZY_MODULES: dict[str, LazyModule] = {
    k: LazyModule(v, alias=k) for k, v in _MODULE_NAMES.items()
}
# Sort the lazy modules per their <module_name>
LAZY_MODULES: dict[str, LazyModule] = dict(
    sorted(LAZY_MODULES.items(), key=lambda x: x[1]._etils_state.module_name)  # pylint: disable=protected-access
)
globals().update(LAZY_MODULES)


__all__ = sorted(LAZY_MODULES)  # Sorted per alias
