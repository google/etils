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

"""Tests for dtypes."""

from __future__ import annotations

import dataclasses
from typing import Any

from etils import array_types
from etils import enp
from etils import epy
from etils.array_types import dtypes
import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Activate the fixture
set_tnp = enp.testing.set_tnp
with_subtests = epy.testing.with_subtests


@dataclasses.dataclass
class _TestItem:
  name: str
  cls_name: str
  asarray_value_dtype: list[tuple[Any, np.dtype]] = dataclasses.field(
      default_factory=list)


_ALL_ITEMS = [
    _TestItem(
        name='AnyDType',
        cls_name='Array',
        asarray_value_dtype=[
            (True, np.bool_),
            ([True, False], np.bool_),
            # Numpy default are plaform dependent (int64 except on windows)
            # (1, np.int32),
            # (1., np.float32),
            # ([1], np.int32),
            # ([1.], np.int32),
            (np.array(1, dtype=np.uint8), np.uint8),
        ],
    ),
    _TestItem(
        name='AnyFloat',
        cls_name='FloatArray',
        asarray_value_dtype=[
            # Bool, int,... casted to float
            (True, np.float32),
            (1, np.float32),
            (1., np.float32),
            ([1], np.float32),
            ([1.], np.float32),
            (np.array([True], dtype=np.bool_), np.float32),
            (np.array(1, dtype=np.uint8), np.float32),
            # Float values not casted
            (np.array(1, dtype=np.float16), np.float16),
            (np.array(1, dtype=jnp.bfloat16), jnp.bfloat16),
        ],
    ),
    _TestItem(
        name='AnyInt',
        cls_name='IntArray',
        asarray_value_dtype=[
            # Casted to int
            (True, np.int32),
            (1, np.int32),
            (1., np.int32),
            ([1], np.int32),
            ([1.], np.int32),
            # Int values not casted
            (np.array([True], dtype=np.bool_), np.int32),
            (np.array(1, dtype=np.uint8), np.uint8),
            # Float values casted
            (np.array(1, dtype=np.float16), np.int32),
        ],
    ),
    # `BoolArray` -> `bool_`
    # _TestItem(
    #     name='bool',
    #     cls_name='BoolArray',
    # ),
    # `StrArray` behavior is undefined for now (and should likely be tested
    # separately)
    # _TestItem(
    #     name='object',
    #     cls_name='StrArray',
    # ),
    _TestItem(
        name='bool',
        cls_name='bool_',
        asarray_value_dtype=[
            (True, np.bool_),
            (1, np.bool_),
            ([1.], np.bool_),
            (np.array([True], dtype=np.bool_), np.bool_),
            (np.array([1], dtype=np.uint8), np.bool_),
        ],
    ),
    _TestItem(
        name='uint8',
        cls_name='ui8',
        asarray_value_dtype=[
            (True, np.uint8),
            (1, np.uint8),
            ([1.], np.uint8),
            (np.array([True], dtype=np.bool_), np.uint8),
            (np.array([1], dtype=np.int32), np.uint8),
        ],
    ),
    _TestItem(
        name='uint16',
        cls_name='ui16',
    ),
    _TestItem(
        name='uint32',
        cls_name='ui32',
    ),
    _TestItem(
        name='uint64',
        cls_name='ui64',
    ),
    _TestItem(
        name='int8',
        cls_name='i8',
    ),
    _TestItem(
        name='int16',
        cls_name='i16',
    ),
    _TestItem(
        name='int32',
        cls_name='i32',
    ),
    _TestItem(
        name='int64',
        cls_name='i64',
    ),
    _TestItem(
        name='float16',
        cls_name='f16',
    ),
    _TestItem(
        name='float32',
        cls_name='f32',
    ),
    _TestItem(
        name='float64',
        cls_name='f64',
    ),
    _TestItem(
        name='complex64',
        cls_name='complex64',
    ),
    _TestItem(
        name='complex128',
        cls_name='complex128',
    ),
]


@pytest.mark.parametrize('item', _ALL_ITEMS)
@enp.testing.parametrize_xnp()
@pytest.mark.usefixtures('with_subtests')
def test_dtype(xnp, item):
  array_type = getattr(array_types, item.cls_name)
  dtype = array_type.dtype
  assert dtype.name == item.name
  assert dtype.array_cls_name == item.cls_name

  # DType casting
  for value, to_dtype in item.asarray_value_dtype:
    with epy.testing.subtest(f'{value}:{to_dtype!r}'):
      array = dtype.asarray(value, xnp=xnp)
      assert isinstance(array, xnp.ndarray)
      assert enp.lazy.as_dtype(array.dtype) == to_dtype


@pytest.mark.parametrize('dtype', [
    np.uint8,
    np.int32,
    np.int64,
    np.float32,
    np.float64,
    np.bool_,
    jnp.bfloat16,
])
@enp.testing.parametrize_xnp()
def test_get_array_dtype_xnp(xnp, dtype):

  # jnp auto-cast float64 -> float32
  assert not jax.config.jax_enable_x64
  target_dtype = dtype
  if xnp is jnp:
    target_dtype = ({
        np.float64: np.float32,
        np.int64: np.int32,
    }).get(dtype, dtype)

  assert dtypes._get_array_dtype(xnp.array(
      [1, 2],
      dtype=dtype,
  )) == target_dtype
