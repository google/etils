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

"""Numpy utils.

Attributes:
  tau: The circle constant (2 * pi). (https://tauday.com/)
"""

import sys
import typing
from typing import Any, TypeVar

from etils.array_types import Array
import numpy as np

_T = TypeVar('_T')

# TODO(pytype): Ideally should use `-> Literal[np]:` but Python does not
# support this: https://github.com/python/typing/issues/1039
# Thankfully, pytype correctly auto-infer `np` when returned by `get_xnp`
NpModule = Any

# Mirror math.tau (PEP 628). See https://tauday.com/
tau = 2 * np.pi


class _LazyImporter:
  """Lazy import module.

  Help to write code seamlessly working with np, Jax and TF.
  Because libs are lazily imported, TF and Jax are always optional dependencies.

  """

  @property
  def has_jax(self) -> bool:
    return 'jax' in sys.modules

  @property
  def has_tf(self) -> bool:
    return 'tensorflow' in sys.modules

  @property
  def jax(self):
    import jax  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error
    return jax

  @property
  def jnp(self):
    import jax.numpy as jnp  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error
    return jnp

  @property
  def tf(self):
    import tensorflow  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error
    return tensorflow

  @property
  def tnp(self):
    import tensorflow.experimental.numpy as tnp  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error
    return tnp

  @property
  def np(self):
    return np

  def is_np(self, x: Array) -> bool:
    return isinstance(x, (np.ndarray, np.generic))

  def is_tf(self, x: Array) -> bool:
    return self.has_tf and isinstance(x, self.tnp.ndarray)

  def is_jax(self, x: Array) -> bool:
    return self.has_jax and isinstance(x, self.jnp.ndarray)

  def is_array(self, x: Array) -> bool:
    return self.is_np(x) or self.is_jax(x) or self.is_tf(x)

  def get_xnp(self, x: Array, *, strict: bool = True):  # -> NpModule:
    """Returns the numpy module associated with the given array.

    Args:
      x: Either tf, jax or numpy array.
      strict: If `False`, default to `np.array` if the array can't be infered (
        to support array-like: list, tuple,...)

    Returns:
      The numpy module.
    """
    # This is inspired from NEP 37 but without the `__array_module__` magic:
    # https://numpy.org/neps/nep-0037-array-module.html
    # Note there is also an implementation of NEP 37 from the author, but look
    # overly complicated and not available at google.
    # https://github.com/seberg/numpy-dispatch
    if self.is_jax(x):
      return self.jnp
    elif self.is_tf(x):
      return self.tnp
    elif self.is_np(x):
      return np
    elif not strict and isinstance(x, (int, bool, float, list, tuple)):
      # `strict=False` support `[0, 0, 0]`, `0`,...
      return np
    else:
      raise TypeError(
          f'Cannot infer the numpy module from array: {type(x).__name__}')


lazy = _LazyImporter()


def get_np_module(array: Array, *, strict: bool = True):  # -> NpModule:
  """Returns the numpy module associated with the given array.

  Args:
    array: Either tf, jax or numpy array.
    strict: If `False`, default to `np.array` if the array can't be infered (
      to support array-like: list, tuple,...)

  Returns:
    The numpy module.
  """
  return lazy.get_xnp(array, strict=strict)


def is_dtype_str(dtype) -> bool:
  """Returns True if the dtype is `str`."""
  # tf.string.as_numpy_dtype is object
  return np.dtype(dtype).type in {np.object_, np.str_, np.bytes_}


def is_array_str(x: Any) -> bool:
  """Returns True if the given array is a `str` array.

  Note: Also returns True for scalar `str`, `bytes` values. For compatibility
  with `tensor.numpy()` which returns `bytes`

  Args:
    x: The array to test

  Returns:
    True or False
  """
  # `Tensor(shape=(), dtype=tf.string).numpy()` returns `bytes`.
  if isinstance(x, (bytes, str)):
    return True
  elif is_array(x):
    return is_dtype_str(x.dtype)
  else:
    return False


def is_array(x: Any) -> bool:
  """Returns `True` if array is np or `jnp` array."""
  if isinstance(x, np.ndarray):
    return True
  elif lazy.has_jax and isinstance(x, lazy.jnp.ndarray):
    return True
  else:
    return False


@np.vectorize
def _to_str_array(x):
  """Decodes bytes -> str array."""
  # tf.string tensors are returned as bytes, so need to convert them back to str
  return x.decode('utf8') if isinstance(x, bytes) else x


@typing.overload
def normalize_bytes2str(x: bytes) -> str:
  ...


@typing.overload
def normalize_bytes2str(x: _T) -> _T:
  ...


# Ideally could also add `BytesArray -> StrArray`, but both `bytes` and `str`
# are `StrArray`
def normalize_bytes2str(x):
  """Normalize `bytes` array to `str` (UTF-8).

  Example of usage:

  ```python
  for ex in tfds.as_numpy(ds):  # tf.data returns `tf.string` as `bytes`
    ex = tf.nest.map_structure(enp.normalize_bytes2str, ex)
  ```

  Args:
    x: Any array

  Returns:
    x: `bytes` array are decoded as `str`
  """
  if isinstance(x, str):
    return x
  if isinstance(x, bytes):
    return x.decode('utf8')
  elif is_array_str(x):
    # Note: `np.char.decode` is likely faster but don't work on `object` nor
    # bytes arrays.
    return _to_str_array(x)
  else:
    return x
