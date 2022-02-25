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

"""Tests for numpy_utils."""

from etils import enp
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow as tf
import tensorflow.experimental.numpy as tnp


# Activate the fixture
set_tnp = enp.testing.set_tnp


def fn(x):
  xnp = enp.get_np_module(x)
  y = xnp.sum(x) + x.mean()
  return x + y


def test_lazy():
  lazy = enp.numpy_utils.lazy

  assert lazy.has_tf
  assert lazy.has_jax

  assert lazy.tf is tf
  assert lazy.jax is jax
  assert lazy.jnp is jnp
  assert lazy.np is np

  assert lazy.is_array(np.array([123]))
  assert lazy.is_np(np.array([123]))
  assert not lazy.is_np(jnp.array([123]))

  assert lazy.is_array(tf.constant([123]))
  assert lazy.is_tf(tf.constant([123]))
  assert not lazy.is_tf(np.array([123]))

  assert lazy.is_array(jnp.array([123]))
  assert lazy.is_jax(jnp.array([123]))
  assert not lazy.is_jax(np.array([123]))

  assert lazy.get_xnp(jnp.array([123])) is jnp
  assert lazy.get_xnp(tf.constant([123])) is tnp
  assert lazy.get_xnp(np.array([123])) is np
  assert lazy.get_xnp([123], strict=False) is np

  with pytest.raises(TypeError, match='Cannot infer the numpy'):
    lazy.get_xnp([123])

  with pytest.raises(TypeError, match='Cannot infer the numpy'):
    lazy.get_xnp(None, strict=False)


@enp.testing.parametrize_xnp()
def test_get_array_module(xnp):
  y = fn(xnp.array([123]))
  assert isinstance(y, xnp.ndarray)


def test_get_array_module_tf():
  y = fn(tf.constant([123]))
  assert isinstance(y, tnp.ndarray)


@pytest.mark.parametrize('xnp', [np, jnp])
def test_not_array_str(xnp):
  x = xnp.array([123])
  assert enp.is_array(x)
  assert not enp.is_array_str(x)
  assert not enp.is_dtype_str(x.dtype)


_STR_DTYPES = [
    np.dtype('<U3'),
    np.dtype('<S3'),
    np.str_,
    np.bytes_,
    str,
    bytes,
    object,
]

_STR_ARRAYS = [
    np.array(['abc', 'def']),
    np.array([b'abc', b'def']),
    np.array(['abc', 'def'], dtype=object),
    np.array([b'abc', b'def'], dtype=object),
]


@pytest.mark.parametrize('array', _STR_ARRAYS)
def test_array_str(array):
  assert enp.is_array(array)
  assert enp.is_array_str(array)
  assert enp.is_dtype_str(array.dtype)


def test_array_str_scalar():
  assert enp.is_array_str('abc')
  assert enp.is_array_str(b'abc')


@pytest.mark.parametrize('dtype', _STR_DTYPES)
def test_is_dtype_str(dtype):
  assert enp.is_dtype_str(dtype)


@pytest.mark.parametrize('dtype', [
    np.dtype(int),
    np.int64,
    int,
])
def test_is_not_dtype_str(dtype):
  assert not enp.is_dtype_str(dtype)


@pytest.mark.parametrize('array', _STR_ARRAYS)
def test_normalize_bytes2str(array):
  assert np.array_equal(
      enp.normalize_bytes2str(array),
      np.array(['abc', 'def']),
  )


def test_normalize_bytes2str_static():
  assert enp.normalize_bytes2str('abc') == 'abc'
  assert enp.normalize_bytes2str(b'abc') == 'abc'
  assert isinstance(enp.normalize_bytes2str('abc'), str)
  assert isinstance(enp.normalize_bytes2str(b'abc'), str)
  assert enp.normalize_bytes2str(123) == 123

  assert np.array_equal(
      enp.normalize_bytes2str(np.array([123, 456])),
      np.array([123, 456]),
  )
  assert isinstance(enp.normalize_bytes2str(jnp.array([1, 2, 3])), jnp.ndarray)
