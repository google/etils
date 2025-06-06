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

"""Tests for interp_utils."""

from etils import enp
import numpy as np


# Activate the fixture
enable_tf_np_mode = enp.testing.set_tnp


@enp.testing.parametrize_xnp()
def test_interp_scalar(xnp: enp.NpModule):
  vals = xnp.asarray(
      [
          [-1, -1],
          [-1, 0],
          [-1, 1],
          [0.5, 1],
          [1, 1],
      ]
  )

  #

  out = enp.interp(vals, from_=(-1, 1), to=(0, 256))
  assert enp.compat.is_array_xnp(out, xnp)

  np.testing.assert_allclose(
      out,
      xnp.asarray([
          [0, 0],
          [0, 128],
          [0, 256],
          [192, 256],
          [256, 256],
      ]),
  )
  np.testing.assert_allclose(
      enp.interp(vals, from_=(-1, 1), to=(0, 1)),
      xnp.asarray([
          [0, 0],
          [0, 0.5],
          [0, 1],
          [0.75, 1],
          [1, 1],
      ]),
  )

  vals = xnp.asarray(
      [
          [255, 255, 0],
          [255, 128, 0],
          [255, 0, 128],
      ]
  )
  np.testing.assert_allclose(
      enp.interp(vals, from_=(0, 255), to=(0, 1)),
      xnp.asarray([
          [1, 1, 0],
          [1, 128 / 255, 0],
          [1, 0, 128 / 255],
      ]),
      atol=1e-6,
  )
  np.testing.assert_allclose(
      enp.interp(vals, from_=(0, 255), to=(-1, 1)),
      xnp.asarray([
          [1, 1, -1],
          [1, 0.00392157, -1],
          [1, -1, 0.00392157],
      ]),
      # np upcast to float64, but jnp keep float32, so reduce precision
      atol=1e-5,
  )


@enp.testing.parametrize_xnp()
def test_interp_coords(xnp):
  coords = xnp.asarray(
      [
          [-1, -1],
          [-1, 0],
          [-1, 1],
          [0.5, 1],
          [1, 1],
      ]
  )
  np.testing.assert_allclose(
      enp.interp(coords, (-1, 1), (0, (1024, 256))),
      xnp.asarray([
          [0, 0],
          [0, 128],
          [0, 256],
          [768, 256],
          [1024, 256],
      ]),
  )

  coords = xnp.asarray(
      [
          [[0, 0], [0, 1024]],
          [[256, 256], [0, 768]],
      ]
  )
  np.testing.assert_allclose(
      enp.interp(coords, (0, (256, 1024)), (0, 1)),
      xnp.asarray([
          [[0, 0], [0, 1]],
          [[1, 0.25], [0, 0.75]],
      ]),
  )
