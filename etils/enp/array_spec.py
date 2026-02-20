# Copyright 2026 The etils Authors.
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

# Is there a way of merging this with array_types ?

from __future__ import annotations

import functools
import sys
from typing import Any, Optional

from etils.enp import numpy_utils
from etils.enp.array_types import typing as array_types
from etils.enp.typing import Array
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
    elif _is_jax_random_dtype(dtype):
      # `jax.random.key(0).dtype` is not a valid np.dtype
      dtype = np.dtype('O')
    self.shape = tuple(shape)
    self.dtype = np.dtype(dtype)

  def __repr__(self) -> str:
    array_type = array_types.ArrayAliasMeta(
        dtype=self.dtype,
        shape=self.shape,
    )
    return repr(array_type)

  def __eq__(self, other) -> bool:
    if not isinstance(other, type(self)):
      return False
    else:
      return (other.shape, other.dtype) == (self.shape, self.dtype)

  def __hash__(self) -> int:
    return hash((self.shape, self.dtype))

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
    # Could refactor with some dynamic registration mechanism.
    if isinstance(array, (np.ndarray, np.generic, ArraySpec)):
      shape = array.shape
      dtype = array.dtype
    elif (
        lazy.has_jax
        and isinstance(array, lazy.jax.Array)
        and lazy.jax.dtypes.issubdtype(array.dtype, lazy.jax.dtypes.prng_key)
    ):
      shape = array.shape
      dtype = np.uint32  # `jax.random.PRNGKeyArray` is a constant
    elif lazy.has_jax and isinstance(
        array,
        (lazy.jax.ShapeDtypeStruct, lazy.jax.Array),
    ):
      shape = array.shape
      dtype = array.dtype
    elif lazy.has_tf and isinstance(
        array,
        (lazy.tf.TensorSpec, lazy.tf.Tensor),
    ):
      shape = array.shape
      # In graph mode, `.shape` values can be `Dimension(32)`
      shape = (int(s) if s is not None else s for s in shape)
      dtype = array.dtype.as_numpy_dtype
    elif lazy.has_tf and isinstance(array, type(_get_none_spec())):
      return None  # Special case for `NoneTensorSpec()`
    elif _is_grain(array):
      shape = array.shape
      dtype = array.dtype
    elif _is_pygrain(array):
      shape = array.shape
      dtype = array.dtype
    elif _is_orbax(array):
      shape = array.shape
      dtype = array.dtype
    elif _is_flax_summary(array):
      shape = array.shape
      dtype = array.dtype
    elif isinstance(array, array_types.ArrayAliasMeta):
      try:
        shape = (int(s) for s in array.shape.split())
      except ValueError:
        raise UnknownArrayError(
            f'Not supported dynamic shape: {array}'
        ) from None
      dtype = array.dtype.np_dtype
    else:
      raise UnknownArrayError(f'Unknown array-like type: {type(array)}')
    # Should we also handle `bytes` case ?
    return cls(shape=shape, dtype=dtype)

  def __konfig_export__(self):
    return {
        '__qualname__': 'etils.enp.array_spec:ArraySpec',
        'shape': self.shape,
        'dtype': self.dtype.name,
    }


def is_fake_array(array: Array) -> bool:
  """Returns `True` if the given array is a fake array."""
  return (
      (lazy.has_jax and isinstance(array, lazy.jax.ShapeDtypeStruct))
      or (lazy.has_tf and isinstance(array, lazy.tf.TensorSpec))
      or isinstance(array, ArraySpec)
      or _is_orbax(array)
      or _is_grain(array)
      or _is_pygrain(array)
      or _is_flax_summary(array)
      or isinstance(array, array_types.ArrayAliasMeta)
  )


def _is_flax_summary(value: Array) -> bool:
  if 'flax.linen' not in sys.modules:
    return False
  from flax import linen as nn  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

  return isinstance(value, nn.summary._ArrayRepresentation)  # pylint: disable=protected-access


def _is_grain(array: Array) -> bool:
  if 'grain.tensorflow' not in sys.modules:
    return False
  from grain import tensorflow as grain  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

  return isinstance(array, grain.ArraySpec)


def _get_grain_shm_array_metadata_cls():
  """Imports the shm metadata from `grain` in a cross-version compatible way."""
  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error
  import grain

  cls = getattr(
      grain.multiprocessing,
      'SharedMemoryArrayMetadata',
      None,
  )
  if cls is None:
    from grain._src.python import shared_memory_array

    cls = shared_memory_array.SharedMemoryArrayMetadata
  # pylint: enable=g-import-not-at-top  # pytype: enable=import-error
  return cls


def _is_pygrain(array: Array) -> bool:
  if (
      'grain._src.python' not in sys.modules
      and 'grain.python' not in sys.modules
  ):
    return False

  return isinstance(array, _get_grain_shm_array_metadata_cls())


def _is_orbax(array: Array) -> bool:
  if 'orbax.checkpoint' not in sys.modules:
    return False
  from orbax.checkpoint.metadata import value  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

  return isinstance(
      array,
      (
          value.ArrayMetadata,
          value.ScalarMetadata,
      ),
  )


def _is_jax_random_dtype(dtype: Any) -> bool:
  return lazy.has_jax and isinstance(dtype, lazy.jax._src.prng.KeyTy)  # pylint: disable=protected-access
