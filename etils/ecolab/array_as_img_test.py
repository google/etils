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

"""Tests for etils.ecolab.array_as_img."""

from unittest import mock

from etils.ecolab import array_as_img
from jax import numpy as jnp
import numpy as np
import pytest
import tensorflow as tf


def test_display_array_as_image():
  with mock.patch('IPython.get_ipython', mock.MagicMock()) as ipython_mock:
    array_as_img.display_array_as_img()
  assert ipython_mock.call_count == 1


@pytest.mark.parametrize('valid_shape', [
    (28, 28),
    (28, 28, 1),
    (28, 28, 3),
])
def test_array_repr_html_valid(valid_shape):
  # 2D images are displayed as images
  assert '<img' in array_as_img._array_repr_html(jnp.zeros(valid_shape))


@pytest.mark.parametrize(
    'invalid_shape',
    [
        (7, 7),
        (28, 7),  # Only one dimension bigger than the threshold
        (28, 28, 4),  # Invalid number of dimension
        (28, 28, 0),
        (2, 28, 28),
    ],
)
def test_array_repr_html_invalid(invalid_shape):
  assert array_as_img._array_repr_html(jnp.zeros(invalid_shape)) is None


@pytest.mark.parametrize('array_cls', [
    tf.constant,
    np.array,
    jnp.array,
])
def test_array_type(array_cls):
  array = array_cls(np.zeros((50, 50)))
  assert '<img' in array_as_img._array_repr_html(array)
