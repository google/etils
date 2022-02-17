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

"""Tests for compat."""

from etils import enp

import numpy as np


@enp.testing.parametrize_xnp()
def test_norm(xnp: enp.NpModule):
  x = xnp.array([3., 0, 0])
  y = enp.compat.norm(x, axis=-1)
  if xnp is np:
    assert isinstance(y, float)
  else:
    assert isinstance(y, xnp.ndarray)
    assert y.shape == ()  # pylint: disable=g-explicit-bool-comparison
  np.testing.assert_allclose(y, 3.)

  y = enp.compat.norm(x, axis=-1, keepdims=True)
  assert isinstance(y, xnp.ndarray)
  assert y.shape == (1,)
  np.testing.assert_allclose(y, [3.])


@enp.testing.parametrize_xnp()
def test_norm_batched(xnp: enp.NpModule):
  x = xnp.array([
      [3., 0, 0],
      [0, 4., 0],
  ])
  y = enp.compat.norm(x, axis=-1)
  assert isinstance(y, xnp.ndarray)
  assert y.shape == (2,)
  np.testing.assert_allclose(y, [3., 4.])

  y = enp.compat.norm(x)
  np.testing.assert_allclose(y, np.sqrt(3**2 + 4**2))
