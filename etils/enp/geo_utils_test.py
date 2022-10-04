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

# Activate the fixture
set_tnp = enp.testing.set_tnp


@enp.testing.parametrize_xnp(with_none=True)
def test_project_onto_plane_vector(xnp: Optional[enp.NpModule]):
  u = [2, 2, 2.0]
  v = [0, 4, 4.0]
  if xnp is not None:
    u = xnp.asarray(u)
    v = xnp.asarray(v)
  else:
    xnp = np
  out = enp.project_onto_vector(u, v)
  assert enp.lazy.get_xnp(out) is xnp
  np.testing.assert_allclose(out, [0, 2, 2], atol=1e-6, rtol=1e-6)

  out = enp.project_onto_plane(u, v)
  assert enp.lazy.get_xnp(out) is xnp
  np.testing.assert_allclose(out, [2, 0, 0], atol=1e-6, rtol=1e-6)
