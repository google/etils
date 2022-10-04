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

"""Common geometric utils."""

from __future__ import annotations

from etils.enp import checking
from etils.enp import linalg
from etils.enp import numpy_utils
from etils.enp.typing import FloatArray


@checking.check_and_normalize_arrays(strict=False)
def project_onto_vector(
    u: FloatArray[3],
    v: FloatArray[3],
    *,
    xnp: numpy_utils.NpModule = ...,
) -> FloatArray[3]:
  """Project `u` onto `v`."""
  return xnp.dot(u, v) / linalg.norm(v) ** 2 * v


@checking.check_and_normalize_arrays(strict=False)
def project_onto_plane(u: FloatArray[3], n: FloatArray[3]) -> FloatArray[3]:
  """Project `u` onto the plane `n` (orthogonal vector)."""
  return u - project_onto_vector(u, n)
