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

from typing import Callable, Iterable, Optional, TypeVar

from etils.enp import numpy_utils
from etils.enp import torch_mock
import numpy as np
import pytest

lazy = numpy_utils.lazy

_FnT = TypeVar('_FnT')


@pytest.fixture(scope='module', autouse=True)
def set_tnp() -> None:
  """Enable numpy behavior (for `tensorflow`).

  Note: The fixture has to be explicitly declared in the `_test.py`
  file where it is used. This can be done by assigning
  `set_tnp = enp.testing.set_tnp`.
  """
  # This is required to have TF follow the same casting rules as numpy
  lazy.tnp.experimental_enable_numpy_behavior(prefer_float32=True)


@pytest.fixture(scope='module', autouse=True)
def set_torch_np() -> None:
  """Enable numpy behavior (for `torch`).

  Note: The fixture has to be explicitly declared in the `_test.py`
  file where it is used. This can be done by assigning
  `set_torch_np = enp.testing.set_torch_np`.
  """
  torch_mock.activate_torch_support()


@pytest.fixture(scope='module', autouse=True)
def enable_torch_tf_np_mode() -> None:
  """Enable numpy behavior (for `torch` and `tensorflow`).

  Note: The fixture has to be explicitly declared in the `_test.py`
  file where it is used. This can be done by assigning
  `enable_torch_tf_np_mode = enp.testing.enable_torch_tf_np_mode`.
  """
  # This is required to have TF follow the same casting rules as numpy
  lazy.tnp.experimental_enable_numpy_behavior(prefer_float32=True)
  torch_mock.activate_torch_support()


def parametrize_xnp(
    *,
    with_none: bool = False,
    restrict: Optional[Iterable[str]] = None,
) -> Callable[[_FnT], _FnT]:
  """Parametrize over the numpy modules.

  Args:
    with_none: If `True`, also yield `None` among the values (to test `list`)
    restrict: If given, only test the given module (e.g. `restrict=['jnp']`)

  Returns:
    The fixture to apply to the `def test_xyz()` function
  """
  name_to_modules = {
      'np': np,
      'jnp': lazy.jnp,
      'tnp': lazy.tnp,
  }

  # Filter modules not requested
  name_to_keep = set(restrict or name_to_modules)
  name_to_modules = {
      k: v for k, v in name_to_modules.items() if k in name_to_keep
  }

  if with_none:
    # Allow to test without numpy module: `x = [1, 2]` vs `x = np.array([1, 2]`
    name_to_modules['no_np'] = None

  return pytest.mark.parametrize(
      'xnp',
      list(name_to_modules.values()),
      ids=list(name_to_modules.keys()),
  )
