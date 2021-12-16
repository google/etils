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

"""Numpy utils."""

import sys
from typing import Any

from etils.array_types import Array
import numpy as np


class _LazyImporter:
  """Lazy import module.

  This allow to use `enp` with numpy only without requiring TF nor Jax.

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


_lazy = _LazyImporter()


def get_np_module(array: Array):  # Ideally should use `-> Literal[np]:``
  """Returns the numpy module associated with the given array.

  Args:
    array: Either tf, jax or numpy array.

  Returns:
    The numpy module.
  """
  # This is inspired from NEP 37 but without the `__array_module__` magic:
  # https://numpy.org/neps/nep-0037-array-module.html
  # Note there is also an implementation of NEP 37 from the author, but look
  # overly complicated and not available at google.
  # https://github.com/seberg/numpy-dispatch
  if isinstance(array, np.ndarray):
    return np
  elif _lazy.has_jax and isinstance(array, _lazy.jnp.ndarray):
    return _lazy.jnp
  elif _lazy.has_tf and isinstance(array, _lazy.tnp.ndarray):
    return _lazy.tnp
  else:
    raise TypeError(
        f'Cannot infer the numpy module from array: {type(array).__name__}')


def is_dtype_str(dtype) -> bool:
  """Returns True if the dtype is `str`."""
  if type(dtype) is object:  # tf.string.as_numpy_dtype is object  # pylint: disable=unidiomatic-typecheck
    return True
  return np.dtype(dtype).kind in {'O', 'S', 'U'}


def is_array_str(x: Array) -> bool:
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
  elif not is_array(x):
    raise TypeError(f'Cannot check `str` on non-array {type(x)}: {x!r}')
  return is_dtype_str(x.dtype)


def is_array(x: Any) -> bool:
  """Returns `True` if array is np or `jnp` array."""
  if isinstance(x, np.ndarray):
    return True
  elif _lazy.has_jax and isinstance(x, _lazy.jnp.ndarray):
    return True
  else:
    return False
