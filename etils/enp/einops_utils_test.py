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

from etils import enp
import numpy as np


def test_flatten_unflatten():
  x = np.ones((2, 3, 4, 2, 1))
  flat_x, batch_shape = enp.flatten(x, '... h w')

  assert flat_x.shape == (2 * 3 * 4, 2, 1)
  assert batch_shape == (2, 3, 4)

  y = enp.unflatten(flat_x, batch_shape, '... h w')

  assert y.shape == (2, 3, 4, 2, 1)


def test_flatten_unflatten_nobatch():
  x = np.ones((3, 2))
  flat_x, batch_shape = enp.flatten(x, '... h w')

  assert flat_x.shape == (1, 3, 2)
  assert batch_shape == ()  # pylint: disable=g-explicit-bool-comparison

  y = enp.unflatten(flat_x, batch_shape, '... h w')

  assert y.shape == (3, 2)
