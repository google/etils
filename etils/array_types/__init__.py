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

"""Typing utils."""

from etils.array_types.typing import ArrayAliasMeta
from etils.array_types.typing import ArrayLike
import numpy as np

Array = ArrayAliasMeta(shape=None, dtype=None)
# TODO(epot): Should have some generic `dtype=float`
FloatArray = ArrayAliasMeta(shape=None, dtype=np.float32)
IntArray = ArrayAliasMeta(shape=None, dtype=np.int32)
BoolArray = ArrayAliasMeta(shape=None, dtype=np.bool_)
StrArray = ArrayAliasMeta(shape=None, dtype=np.dtype('O'))

f32 = ArrayAliasMeta(shape=None, dtype=np.float32)
ui8 = ArrayAliasMeta(shape=None, dtype=np.uint8)
ui32 = ArrayAliasMeta(shape=None, dtype=np.uint32)
i32 = ArrayAliasMeta(shape=None, dtype=np.int32)
bool_ = ArrayAliasMeta(shape=None, dtype=np.bool_)

# Random number generator jax key
PRNGKey = ui32[2]

# Keep API clean
del np

__all__ = [
    'ArrayAliasMeta',
    'ArrayLike',
    'Array',
    'FloatArray',
    'IntArray',
    'BoolArray',
    'StrArray',
    'f32',
    'ui8',
    'ui32',
    'i32',
    'bool_',
    'PRNGKey',
]
