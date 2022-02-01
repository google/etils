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

"""Tests for array_spec."""

from etils import enp
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf


def _array_repr(x) -> str:
  return repr(enp.ArraySpec.from_array(x))


def test_array_spec_valid():
  assert enp.ArraySpec((), np.float32) == enp.ArraySpec([], np.dtype('float32'))

  assert not enp.ArraySpec.is_array(123)
  assert not enp.ArraySpec.is_array('123')
  assert enp.ArraySpec.is_array(np.array(123))
  assert enp.ArraySpec.is_array(np.array('123'))


def test_array_spec_tensors():
  # ====== Numpy ======
  # np.array
  assert _array_repr(np.zeros((3,), dtype=np.float64)) == 'f64[3]'
  assert _array_repr(np.array(123)) == 'int64[]'
  # str arrays
  assert _array_repr(np.array('123')) == 'str[]'
  assert _array_repr(np.array(['abc', 'def', ''], dtype=object)) == 'str[3]'
  assert _array_repr(np.array(['abc', 'def'])) == 'str[2]'

  # ====== Jax ======
  # jnp.array
  assert _array_repr(jnp.ones((5,), dtype=jnp.bool_)) == 'bool_[5]'
  # jax.ShapeDtypeStruct
  assert _array_repr(jax.ShapeDtypeStruct((6,), dtype=np.int32)) == 'i32[6]'

  # ====== TensorFlow ======
  # tf.Tensor
  assert _array_repr(tf.zeros((3,))) == 'f32[3]'
  # tf.TensorSpec
  assert _array_repr(tf.TensorSpec((None,), dtype=tf.int32)) == 'i32[_]'
  assert _array_repr(tf.TensorSpec((None, 3), dtype=tf.int32)) == 'i32[_ 3]'
  # str tensors
  assert _array_repr(tf.TensorSpec((4,), dtype=tf.string)) == 'str[4]'
  # tf.NoneTensorSpec
  assert enp.ArraySpec.from_array(enp.array_spec._get_none_spec()) is None

  # ====== Etils ======
  assert _array_repr(enp.ArraySpec((1, 2), dtype=np.float32)) == 'f32[1 2]'
  assert _array_repr(enp.ArraySpec((None, 2), dtype=str)) == 'str[_ 2]'
  # TODO(epot): support array_types ?


def test_array_spec_repr():
  assert repr(enp.ArraySpec((), np.float32)) == 'f32[]'
  assert repr(enp.ArraySpec((1, 3), np.uint8)) == 'ui8[1 3]'
  assert repr(enp.ArraySpec((), np.complex64)) == 'complex64[]'
  assert repr(enp.ArraySpec((4,), np.dtype('O'))) == 'str[4]'
  assert repr(enp.ArraySpec((4,), str)) == 'str[4]'
  # `str()` works too:
  assert str(enp.ArraySpec((1,), np.int32)) == 'i32[1]'
