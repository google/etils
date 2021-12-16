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
  elif 'jax' in sys.modules and _is_jax_array(x):
    return True
  else:
    return False


def _is_jax_array(x):
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error
  return isinstance(x, jnp.ndarray)
