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


@enp.testing.parametrize_xnp(with_none=True)
@pytest.mark.parametrize('u_shape', [(3,), (2, 1, 3)])
@pytest.mark.parametrize('v_shape', [(3,)])  # TODO(epot): Support `(2, 1, 3)`
def test_project_onto_plane_vector(
    xnp: Optional[enp.NpModule], u_shape, v_shape
):
  expected_shape = np.broadcast_shapes(u_shape, v_shape)

  u = [2, 2, 2.0]
  v = [0, 4, 4.0]
  if xnp is not None:
    u = xnp.asarray(u)
    v = xnp.asarray(v)
    u = xnp.broadcast_to(u, u_shape)
    v = xnp.broadcast_to(v, v_shape)
  else:
    xnp = np
    if expected_shape != (3,):  # pylint: disable=g-explicit-bool-comparison
      return

  out = enp.project_onto_vector(u, v)
  assert enp.lazy.get_xnp(out) is xnp
  assert out.shape == expected_shape
  expected_out = np.broadcast_to([0, 2, 2], expected_shape)
  np.testing.assert_allclose(out, expected_out, atol=1e-6, rtol=1e-6)

  out = enp.project_onto_plane(u, v)
  assert enp.lazy.get_xnp(out) is xnp
  assert out.shape == expected_shape
  expected_out = np.broadcast_to([2, 0, 0], expected_shape)
  np.testing.assert_allclose(out, expected_out, atol=1e-6, rtol=1e-6)
