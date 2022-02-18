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

"""Compat module for interoperability between tf, jax, numpy."""

from typing import Optional

from etils.array_types import FloatArray
from etils.enp import numpy_utils

lazy = numpy_utils.lazy


def norm(
    x: FloatArray['*d'],
    axis: Optional[int] = None,
    keepdims: bool = False,
) -> FloatArray['*d']:
  """Like `np.linalg.norm` but auto-support jnp, tnp, np."""
  if lazy.is_tf(x):  # TODO(b/219427516): tnp.linalg.norm missing
    return lazy.tf.norm(x, axis=axis, keepdims=keepdims)
  xnp = lazy.get_xnp(x)
  return xnp.linalg.norm(x, axis=axis, keepdims=keepdims)


def inv(x: FloatArray['*d']) -> FloatArray['*d']:
  """Like `np.linalg.inv` but auto-support jnp, tnp, np."""
  if lazy.is_tf(x):  # TF Compatibility
    return lazy.tf.linalg.inv(x)
  else:
    return lazy.get_xnp(x).linalg.inv(x)
