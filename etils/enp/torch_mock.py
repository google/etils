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

"""Torch compatibility."""

from __future__ import annotations

import functools
import typing

from etils.enp import numpy_utils
import numpy as np

if typing.TYPE_CHECKING:
  import torch as torch_  # pytype: disable=import-error

lazy = numpy_utils.lazy


def activate_torch_support() -> None:
  """Activate numpy behavior for `torch`.

  This function mocks `torch` to make its behavior closer to `numpy` by:

  *   Adding some missing methods (`torch.ndarray`, `torch.expand_dims`,...)
  *   Mocking a few methods to support `np.dtype` (
      https://github.com/pytorch/pytorch/issues/40568)
  """
  torch = lazy.torch
  if hasattr(torch, '__etils_np_mode__'):  # Already mocked
    return
  _mock_torch(torch)
  torch.__etils_np_mode__ = True


@functools.lru_cache()
def _torch_to_np_dtypes() -> dict[torch_.dtype, np.dtype]:
  """Returns mapping torch -> numpy dtypes."""
  torch = lazy.torch
  return {
      torch.bool: np.bool_,
      torch.uint8: np.uint8,
      torch.int8: np.int8,
      torch.int16: np.int16,
      torch.int32: np.int32,
      torch.int64: np.int64,
      # TODO(epot): torch.bfloat:
      torch.float16: np.float16,
      torch.float32: np.float32,
      torch.float64: np.float64,
      torch.complex64: np.complex64,
      torch.complex128: np.complex128,
  }


@functools.lru_cache()
def _np_to_torch_dtypes() -> dict[np.dtype, torch_.dtype]:
  """Returns mapping numpy -> torch dtypes."""
  return dict((np.dtype(n), t) for t, n in _torch_to_np_dtypes().items())


def dtype_torch_to_np(dtype) -> np.dtype:
  """Returns the numpy dtype for the given torch dtype."""
  return _torch_to_np_dtypes()[dtype]


def _mock_torch(torch) -> None:
  """Mock `torch` to behave more like `numpy`."""
  # Mock a few pytorch functions to make them behave like numpy/jnp/tf.numpy
  # * Accept `dtype=np.int32` (by default, `torch` only accept torch dtype)
  # * More flexible casting (`Tensor(tf.int32).mean()` returns `tf.float32`,
  #   `torch.allclose()` works on different input types)
  # * Accept `torch.zeros(shape=)` (currently only `size=` accepted)
  for fn_name in [
      'tensor',
      'asarray',
      'zeros',
      'ones',
  ]:
    _wrap_fn(torch, fn_name, _cast_dtype_kwargs)
  _wrap_fn(torch.Tensor, 'type', _cast_dtype_arg)
  _wrap_fn(torch.Tensor, 'mean', _mean)


def _wrap_fn(obj, name: str, fn) -> None:
  """Replace the function by the new one."""
  original_fn = getattr(obj, name)

  @functools.wraps(original_fn)
  def new_fn(*args, **kwargs):
    return fn(original_fn, *args, **kwargs)

  setattr(obj, name, new_fn)


def _to_torch_dtype(dtype):
  if dtype is ... or dtype is None or lazy.is_torch_dtype(dtype):
    return dtype
  else:
    return _np_to_torch_dtypes()[np.dtype(dtype)]


def _cast_dtype_kwargs(fn, *args, dtype=..., **kwargs):
  """Normalize dtype (passed as kwargs)."""
  dtype = _to_torch_dtype(dtype)
  if dtype is ...:  # Use `...` as sentinel value
    return fn(*args, **kwargs)
  else:
    return fn(*args, **kwargs, dtype=dtype)


def _cast_dtype_arg(fn, self, dtype=..., **kwargs):
  """Normalize dtype (passed as args)."""
  dtype = _to_torch_dtype(dtype)
  if dtype is ...:
    return fn(self, **kwargs)
  else:
    return fn(self, dtype, **kwargs)


def _mean(fn, self: torch_.Tensor, *args, **kwargs):
  """`x.mean()` return `float32` for `intXX` arrays."""
  if not self.dtype.is_floating_point:
    return fn(self, *args, **kwargs, dtype=lazy.torch.float32)
  else:
    return fn(self, *args, **kwargs)
