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

"""Tests for checking."""

from etils import enp
import numpy as np
import pytest


# Activate fixture
@pytest.fixture(scope='module', autouse=True)
def set_tnp() -> None:
  """Enable numpy behavior.

  Note: The fixture has to be explicitly declared in the `_test.py`
  file where it is used. This can be done by assigning
  `set_tnp = enp.testing.set_tnp`.
  """
  import tensorflow.experimental.numpy as tnp
  # This is required to have TF follow the same casting rules as numpy
  tnp.experimental_enable_numpy_behavior(prefer_float32=True)


@enp.testing.parametrize_xnp()
@pytest.mark.parametrize('fn', [1, 2, 3])
def test_type(xnp: enp.NpModule, fn):
  x = xnp.array([2.0], dtype=np.float32)
  y = xnp.array([1], dtype=np.int32)
