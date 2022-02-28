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

"""Test utils."""

from typing import Callable, TypeVar

from etils.enp import numpy_utils
import numpy as np
import pytest

lazy = numpy_utils.lazy

_FnT = TypeVar('_FnT')


@pytest.fixture(scope='module', autouse=True)
def set_tnp() -> None:
  """Enable numpy behavior.

  Note: The fixture has to be explicitly declared in the `_test.py`
  file where it is used. This can be done by assigning
  `set_tnp = testing.set_tnp`.

  """
  # This is required to have TF follow the same casting rules as numpy
  lazy.tnp.experimental_enable_numpy_behavior(prefer_float32=True)


def parametrize_xnp(*, with_none: bool = False) -> Callable[[_FnT], _FnT]:
  """Parametrize over the numpy modules."""
  np_modules = [np, lazy.jnp, lazy.tnp]
  np_names = ['np', 'jax', 'tf']
  if with_none:
    # Allow to test without numpy module: `x = [1, 2]` vs `x = np.array([1, 2]`
    np_modules.append(None)
    np_names.append('no_np')
  return pytest.mark.parametrize('xnp', np_modules, ids=np_names)
