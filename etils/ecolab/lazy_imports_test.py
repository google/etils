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

"""Tests for lazy_imports."""

import sys

from etils import ecolab

# We cannot globally import inside test:
#
# `from etils.ecolab.lazy_imports import *`
#
# Indeed, `pytest` explore `dir(lazy_imports_test.xyz)` to collect tests and is
# checking for various attributes (e.g., like `_pytestfixturefunction`) which
# has the side effect of triggering all imports


def test_lazy_imports():
  from etils.ecolab.lazy_imports import jax  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

  assert repr(jax) == "LazyModule('jax')"

  assert jax.numpy.zeros((2, 3)).shape == (2, 3)
  assert "jax" in sys.modules
  assert repr(jax).startswith("<lazy_module 'jax'")
  assert jax._etils_state.module_loaded

  from etils.ecolab.lazy_imports import jnp  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

  assert not jnp._etils_state.module_loaded
  _ = jnp.array  # Trigger import
  assert jnp._etils_state.module_loaded
  del jnp

  from etils.ecolab.lazy_imports import epy  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

  _ = epy.reraise  # Trigger import

  # Import os but do not trigger imports
  from etils.ecolab.lazy_imports import os  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

  del os

  # lazy_imports can be imported but do not appear in `print_current_imports`
  from etils.ecolab.lazy_imports import lazy_imports  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

  assert not lazy_imports._etils_state.module_loaded
  _ = lazy_imports.print_current_imports  # Trigger import
  assert lazy_imports._etils_state.module_loaded
  del lazy_imports

  assert ecolab.lazy_imports._current_import_statements() == epy.dedent(
      """
      from etils import epy
      import jax
      import jax.numpy as jnp
      """
  )


def test_lazy_imports_mutate():
  from etils.lazy_imports import enp  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

  assert not enp._etils_state.module_loaded
  _ = enp.NpModule
  assert enp._etils_state.module_loaded

  enp.some_value = 123

  # Mutating lazy import is propagated to the shared instance.
  from etils import enp as enp2  # pylint: disable=g-import-not-at-top

  assert enp2.some_value == 123  # pytype: disable=module-attr


def test_lazy_imports_built_in():
  from etils.ecolab.lazy_imports import gc  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

  assert repr(gc).startswith("LazyModule('gc')")
  _ = gc.collect
  assert repr(gc).startswith("<lazy_module 'gc'")


def test_lazy_imports_dir():
  from etils.ecolab.lazy_imports import lazy_imports  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

  for attr_name in dir(lazy_imports):
    assert hasattr(lazy_imports, attr_name)
