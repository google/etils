# Copyright 2024 The etils Authors.
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

# Note: explicitly *not* using `enable_tf_np_mode = enp.testing.set_tnp` in this
# test, to ensure that the `compat` module works in non-numpy-tf-behavior mode.


@enp.testing.parametrize_xnp()
def test_dtype_convert(xnp: enp.NpModule):
  bools = xnp.asarray([True, False, True, False])
  floats = enp.compat.astype(bools, xnp.float32)
  assert floats.dtype == xnp.float32
  np.testing.assert_allclose(floats, [1.0, 0.0, 1.0, 0.0])
