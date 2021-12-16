# Copyright 2021 The etils Authors.
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

"""Tests for numpy_utils."""

from etils import enp
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow as tf
import tensorflow.experimental.numpy as tnp


@pytest.fixture(scope='module', autouse=True)
def setup_tf():
  tnp.experimental_enable_numpy_behavior()


def fn(x):
  xnp = enp.get_np_module(x)
  y = xnp.sum(x) + x.mean()
  return x + y


def test_get_array_module_tf():
  y = fn(tf.constant([123]))
  assert isinstance(y, tnp.ndarray)


@pytest.mark.parametrize('np_module', [np, jnp, tnp])
def test_get_array_module(np_module):
  y = fn(np_module.array([123]))
  assert isinstance(y, np_module.ndarray)


@pytest.mark.parametrize('np_module', [np, jnp])
def test_not_array_str(np_module):
  x = np_module.array([123])
  assert enp.is_array(x)
  assert not enp.is_array_str(x)
  assert not enp.is_dtype_str(x.dtype)


def test_array_str():
  x = np.array(['abc'])
  assert enp.is_array(x)
  assert enp.is_array_str('abc')
  assert enp.is_array_str(b'abc')
  assert enp.is_array_str(x)
  assert enp.is_dtype_str(x.dtype)
