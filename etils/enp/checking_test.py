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

"""Tests for checking."""

from etils import enp
from etils.array_types import IntArray, FloatArray, f32  # pylint: disable=g-multiple-import
import numpy as np
import pytest

# Activate fixture
enable_tf_np_mode = enp.testing.set_tnp


@enp.check_and_normalize_arrays
def fn_base(x: f32, y: IntArray):
  assert enp.lazy.get_xnp(x) is enp.lazy.get_xnp(y)
  return x + y


@enp.check_and_normalize_arrays(strict=False)
def fn_non_strict(x: f32, y: IntArray):
  assert enp.lazy.get_xnp(x) is enp.lazy.get_xnp(y)
  return x + y


@enp.check_and_normalize_arrays
def fn_xnp_kwarg(x: f32, y: IntArray, *, xnp: enp.NpModule = ...):
  assert enp.lazy.get_xnp(x) is enp.lazy.get_xnp(y)
  assert enp.lazy.get_xnp(x) is xnp
  return x + y


def _assert_out(z, xnp):
  assert enp.compat.is_array_xnp(z, xnp)
  # jnp/np don't have same upcasting rules
  assert enp.lazy.as_dtype(z.dtype) in (np.float32, np.float64)
  assert z.shape == (1,)
  assert z[0] == 3.0


@enp.testing.parametrize_xnp()
@pytest.mark.parametrize('fn', [fn_base, fn_xnp_kwarg, fn_non_strict])
def test_type(xnp: enp.NpModule, fn):
  x = xnp.asarray([2.0], dtype=xnp.float32)
  y = xnp.asarray([1], dtype=xnp.int32)

  _assert_out(fn(x, y), xnp)
  # IntArray accept any int dtype
  _assert_out(fn(x, xnp.asarray([1], dtype=xnp.uint8)), xnp)

  if fn is fn_non_strict:
    # strict=False and pass non-array
    _assert_out(fn(x, [1]), xnp)
    _assert_out(fn([2.0], y), xnp)
    _assert_out(fn([2], y), xnp)  # auto int -> float conversion
    _assert_out(fn([2.0], [1]), np)  # Default to np
  else:
    with pytest.raises(TypeError, match='Expected xnp.ndarray'):
      fn(x, [1])

  # Bad dtype
  with pytest.raises(ValueError, match='Cannot cast float16 to float32'):
    fn(xnp.asarray(2.0, dtype=xnp.float16), y)

  # Bad dtype (integer)
  with pytest.raises(ValueError, match='Cannot cast float32 to'):
    fn(x, xnp.asarray(2.0, dtype=xnp.float32))

  # Independently of the original xnp, we can explicitly pass the target xnp
  _assert_out(fn(x, y, xnp=enp.lazy.np), enp.lazy.np)  # pytype: disable=wrong-keyword-args
  _assert_out(fn(x, y, xnp=enp.lazy.jnp), enp.lazy.jnp)  # pytype: disable=wrong-keyword-args
  _assert_out(fn(x, y, xnp=enp.lazy.tnp), enp.lazy.tnp)  # pytype: disable=wrong-keyword-args
  # TODO(epot): `torch.asarray` do not work with `tf` / `jax`
  # _assert_out(fn(x, y, xnp=enp.lazy.torch), enp.lazy.torch)  # pytype: disable=wrong-keyword-args

  # Pass a xnp and np yield xnp
  _assert_out(fn(x, np.asarray(y)), xnp)

  # Raise an error when mixing both jnp and TF
  with pytest.raises(ValueError, match='Conflicting numpy types'):
    fn(enp.lazy.jnp.asarray([2.0]), enp.lazy.tnp.asarray([1]))


def test_non_array_annotations():
  @enp.check_and_normalize_arrays(strict=False)
  def fn_non_array_args(x: int, y: FloatArray, z):
    # Non-array typing annotations are preserved
    assert isinstance(x, int)
    assert isinstance(z, str)
    assert enp.lazy.get_xnp(y) is enp.lazy.jnp
    return y + x

  _assert_out(fn_non_array_args(1, [2], 'abc', xnp=enp.lazy.jnp), enp.lazy.jnp)  # pytype: disable=wrong-keyword-args


def test_missing_xnp_default():
  @enp.check_and_normalize_arrays(strict=False)
  def fn_missing_default(x: FloatArray, *, xnp: enp.NpModule):
    del xnp
    return x

  fn_missing_default(1.0)  # pytype: disable=missing-parameter  # pylint: disable=missing-kwoa
