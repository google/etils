# Copyright 2025 The etils Authors.
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

"""Tests for linalg compat module."""

from etils import enp

import numpy as np

enable_tf_np_mode = enp.testing.set_tnp


@enp.testing.parametrize_xnp()
def test_normalize(xnp: enp.NpModule):
  x = xnp.asarray([3.0, 0, 0])
  y = enp.linalg.normalize(x)
  assert enp.compat.is_array_xnp(y, xnp)
  assert y.shape == x.shape
  np.testing.assert_allclose(y, [1.0, 0.0, 0.0])


@enp.testing.parametrize_xnp()
def test_normalize_batched(xnp: enp.NpModule):
  x = xnp.asarray(
      [
          [3.0, 0, 0],
          [0, 4.0, 0],
          [2.0, 3.0, 0],
      ]
  )
  y = enp.linalg.normalize(x)
  assert enp.compat.is_array_xnp(y, xnp)
  assert y.shape == x.shape
  norm = np.sqrt(2**2 + 3**2)
  np.testing.assert_allclose(
      y,
      [
          [1.0, 0, 0],
          [0, 1.0, 0],
          [2.0 / norm, 3.0 / norm, 0],
      ],
  )


@enp.testing.parametrize_xnp()
def test_norm(xnp: enp.NpModule):
  x = xnp.asarray([3.0, 0, 0])
  y = enp.compat.norm(x, axis=-1)
  if xnp is np:
    assert isinstance(y, float)
  else:
    assert enp.compat.is_array_xnp(y, xnp)
    assert y.shape == ()  # pylint: disable=g-explicit-bool-comparison
  np.testing.assert_allclose(y, 3.0)

  y = enp.compat.norm(x, axis=-1, keepdims=True)
  assert enp.compat.is_array_xnp(y, xnp)
  assert y.shape == (1,)
  np.testing.assert_allclose(y, [3.0])


@enp.testing.parametrize_xnp()
def test_norm_batched(xnp: enp.NpModule):
  x = xnp.asarray(
      [
          [3.0, 0, 0],
          [0, 4.0, 0],
      ]
  )
  y = enp.compat.norm(x, axis=-1)
  assert enp.compat.is_array_xnp(y, xnp)
  assert y.shape == (2,)
  np.testing.assert_allclose(y, [3.0, 4.0])

  y = enp.compat.norm(x)
  np.testing.assert_allclose(y, np.sqrt(3**2 + 4**2))
