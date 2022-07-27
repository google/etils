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

"""Tests for typing."""

from etils import enp
from etils.array_types import Array, f32, ui8  # pylint: disable=g-multiple-import
import jax.numpy as jnp
import numpy as np
import pytest

# TODO(epot): Add `bfloat16` to array_types. Not this might require some
# LazyDType to lazy-load jax.
bf16 = enp.typing.ArrayAliasMeta(shape=None, dtype=np.dtype(jnp.bfloat16))


# Make sure annotations do not trigger pytype errors
def f(x: f32, y: ui8['... c']):
  return x + y


def g():
  x0: Array = np.array([0])
  x1: Array['...'] = np.array([0])
  x2: f32['...'] = np.array([0])
  x3: ui8['...'] = np.array([0])

  f(np.array([0]), np.array([0]))
  f(x0, x1)
  f(x2, x3)


@pytest.mark.parametrize(
    'alias, repr_, shape, dtype',
    [
        (Array, 'Array[...]', '...', None),  # No dtype/shape defined
        (Array[''], 'Array[]', '', None),
        (Array['h w c'], 'Array[h w c]', 'h w c', None),
        (f32, 'f32[...]', '...', np.float32),  # float32
        (f32['x'], 'f32[x]', 'x', np.float32),
        (ui8['h w'], 'ui8[h w]', 'h w', np.uint8),
        (f32[1], 'f32[1]', '1', np.float32),  # int
        (f32[()], 'f32[]', '', np.float32),  # tuple
        (f32[1, 3], 'f32[1 3]', '1 3', np.float32),  # tuple[int]
        (f32[4, 'h', 'w'], 'f32[4 h w]', '4 h w', np.float32),  # tuple[str]
        (f32[...], 'f32[...]', '...', np.float32),  # With elipsis
        (f32[..., 3], 'f32[... 3]', '... 3', np.float32),
        (bf16[..., 3], 'bf16[... 3]', '... 3', np.dtype(jnp.bfloat16)),
    ],
)
def test_array_alias(alias, repr_, shape, dtype):
  assert repr(alias) == repr_
  assert str(alias) == repr_
  assert alias.shape == shape
  assert isinstance(alias.dtype, enp.dtypes.DType)
  assert alias.__name__ == alias.dtype.array_cls_name
  assert alias.dtype == enp.dtypes.DType.from_value(dtype)


def test_array_dtype():
  assert f32['h w'].dtype == f32['h', 'w'].dtype
  assert hash(f32['h w'].dtype) == hash(f32['h', 'w'].dtype)


def test_array_eq():
  assert f32['h w'] == f32['h', 'w']
  assert f32['1 2 3'] == f32[1, 2, 3]
  assert f32[...] == f32['...']

  assert f32[None] == f32['_']
  assert f32[None, 3] == f32['_ 3']
  assert f32[None, None] == f32['_ _']
  assert f32[..., None, 3] == f32['... _ 3']
  assert f32['... _', 3] == f32['... _ 3']

  assert f32['h w'] != f32['h c']
  assert f32['h w'] != Array['h w']
  assert f32['h w'] != ui8['h w']

  assert {f32['h w'], f32['h w'], f32['h', 'w']} == {f32['h w']}
