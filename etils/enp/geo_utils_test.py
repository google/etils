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

"""Tests for geo_utils."""

from typing import Optional

from etils import enp
import numpy as np
import pytest

# Activate the fixture
set_tnp = enp.testing.set_tnp


def _array(xnp, value, batch_shape):
  x = xnp.asarray(value)
  x = xnp.broadcast_to(x, batch_shape + x.shape)
  return x


def _assert_equal(x0, x1, xnp):
  assert enp.lazy.get_xnp(x0) is xnp
  np.testing.assert_allclose(x0, x1, atol=1e-6, rtol=1e-6)


@enp.testing.parametrize_xnp(restrict=['np'])
@pytest.mark.parametrize(
    'u, v, expected',
    [
        ([0, 0, 1.0], [0, 10, 0.0], 1 / 4 * enp.tau),
        ([0, 0, 1.0], [0, 0, 2.0], 0.0),
        ([0, 0, 1.0], [0, 0, -2.0], 1 / 2 * enp.tau),
        ([0, 2, 2.0], [0, 0, 1.0], 1 / 8 * enp.tau),
    ],
)
@pytest.mark.parametrize('u_shape', [(), (2, 1)])
@pytest.mark.parametrize('v_shape', [(), (2, 1)])
def test_angle_between(
    xnp: enp.NpModule,
    u,
    v,
    expected,
    u_shape,
    v_shape,
):
  u = _array(xnp, u, u_shape)
  v = _array(xnp, v, v_shape)
  expected_shape = np.broadcast_shapes(u_shape, v_shape)
  expected = _array(xnp, expected, expected_shape)

  _assert_equal(enp.angle_between(u, v), expected, xnp)


@enp.testing.parametrize_xnp()
@pytest.mark.parametrize('u_shape', [(), (2, 1)])
@pytest.mark.parametrize('v_shape', [(), (2, 1)])
def test_project_onto_plane_vector(
    xnp: Optional[enp.NpModule], u_shape, v_shape
):
  expected_shape = np.broadcast_shapes(u_shape, v_shape)

  u = _array(xnp, [2, 2, 2.0], u_shape)
  v = _array(xnp, [0, 4, 4.0], v_shape)

  expected = _array(xnp, [0, 2, 2], expected_shape)
  _assert_equal(enp.project_onto_vector(u, v), expected, xnp)

  expected = _array(xnp, [2, 0, 0], expected_shape)
  _assert_equal(enp.project_onto_plane(u, v), expected, xnp)
