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

"""Array spec utils."""

from __future__ import annotations

import functools
from typing import Any, Optional

from etils import array_types
from etils.array_types import Array
from etils.enp import numpy_utils
import numpy as np

lazy = numpy_utils.lazy


class UnknownArrayError(TypeError):
  pass


@functools.lru_cache()
def _get_none_spec():  # -> tf.TypeSpec:
  """Returns the tf.NoneTensorSpec()."""
  assert lazy.has_tf
  # We need this hack as NoneTensorSpec is not exposed in the public API.
  # (see: b/191132147)
  ds = lazy.tf.data.Dataset.range(0)
  ds = ds.map(lambda x: (x, None))
  return ds.element_spec[-1]


class ArraySpec:
  """Structure containing shape/dtype."""

  __slots__ = ['shape', 'dtype']

  def __init__(self, shape, dtype):
    if numpy_utils.is_dtype_str(dtype):  # Normalize `str` dtype
      dtype = np.dtype('O')
    self.shape = tuple(shape)
    self.dtype = np.dtype(dtype)

  def __repr__(self) -> str:
    shape_str = ' '.join(['_' if s is None else str(s) for s in self.shape])
    dtype_str = array_types.typing.DTYPE_NP_TO_COMPACT_STR.get(
        self.dtype, self.dtype.name)
    return f'{dtype_str}[{shape_str}]'

  def __eq__(self, other) -> bool:
    if not isinstance(other, type(self)):
      return False
    else:
      return (other.shape, other.dtype) == (self.shape, self.dtype)

  @classmethod
  def is_array(cls, array: Any) -> bool:
    """Returns `True` if the given value can be converted to `ArraySpec`."""
    try:
      cls.from_array(array)
    except UnknownArrayError:
      return False
    else:
      return True

  @classmethod
  def from_array(cls, array: Array) -> Optional[ArraySpec]:
    """Construct the `ArraySpec` from the given array."""
    if isinstance(array, (np.ndarray, np.generic, ArraySpec)):
      shape = array.shape
      dtype = array.dtype
    elif lazy.has_jax and isinstance(
        array,
        (lazy.jax.ShapeDtypeStruct, lazy.jnp.ndarray),
    ):
      shape = array.shape
      dtype = array.dtype
    elif lazy.has_tf and isinstance(
        array,
        (lazy.tf.TensorSpec, lazy.tf.Tensor),
    ):
      shape = array.shape
      dtype = array.dtype.as_numpy_dtype
    elif lazy.has_tf and isinstance(array, type(_get_none_spec())):
      return None  # Special case for `NoneTensorSpec()`
    else:
      raise UnknownArrayError(f'Unknown array-like type: {array!r}')
    # Should we also handle `bytes` case ?
    return cls(shape=shape, dtype=dtype)
