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
from typing import Any, Tuple, Union

from etils import array_types
from etils import enp
from etils import epy
import jax.numpy as jnp
import numpy as np
import pytest

# Activate the fixture
set_tnp = enp.testing.set_tnp
with_subtests = epy.testing.with_subtests

_ArrayDtype = Tuple[Any, Union[np.dtype, Exception]]


@dataclasses.dataclass
class _ArrayItem:
  """Test that `asarray(value).dtype` is valid.

  Attributes:
    value: array or array-like
    dtype: Expected dtype, after `dtype.asarray(value)` conversion
    iscast: If `True`, `dtype.asarray(value, casting='none')` should raise an
      error. (as implicit casting is done)
  """

  value: Any
  dtype: np.dtype
  iscast: bool = False


@dataclasses.dataclass
class _DTypeTestItem:
  name: str
  cls_name: str
  array_items: list[_ArrayItem] = dataclasses.field(default_factory=list)


# TODO(epot): Better inference of array-like dtype
# Replace `, iscast=False` by `=True`

_ALL_ITEMS = [
    _DTypeTestItem(
        name='AnyDType',
        cls_name='Array',
        array_items=[
            _ArrayItem(True, np.bool_),
            _ArrayItem([True, False], np.bool_),
            # Numpy default are plaform dependent (int64 except on windows)
            # _ArrayItem(1, np.int32),
            # _ArrayItem(1., np.float32),
            # _ArrayItem([1], np.int32),
            # _ArrayItem([1.], np.int32),
            _ArrayItem(np.array(1, dtype=np.uint8), np.uint8),
        ],
    ),
    _DTypeTestItem(
        name='AnyFloat',
        cls_name='FloatArray',
        array_items=[
            # Bool, int,... casted to float
            _ArrayItem(True, np.float32, iscast=True),
            _ArrayItem(1, np.float32, iscast=False),
            _ArrayItem(1.0, np.float32),
            _ArrayItem([1], np.float32, iscast=False),
            _ArrayItem([1.0], np.float32),
            _ArrayItem(
                np.array([True], dtype=np.bool_), np.float32, iscast=True
            ),
            _ArrayItem(np.array(1, dtype=np.uint8), np.float32, iscast=True),
            # Float values not casted
            _ArrayItem(np.array(1, dtype=np.float16), np.float16),
            _ArrayItem(np.array(1, dtype=jnp.bfloat16), jnp.bfloat16),
        ],
    ),
    _DTypeTestItem(
        name='AnyInt',
        cls_name='IntArray',
        array_items=[
            # Casted to int
            _ArrayItem(True, np.int32, iscast=True),
            _ArrayItem(1, np.int32),
            _ArrayItem(1.0, np.int32, iscast=False),
            _ArrayItem([1], np.int32),
            _ArrayItem([1.0], np.int32, iscast=False),
            # Int values not casted
            _ArrayItem(np.array([True], dtype=np.bool_), np.int32, iscast=True),
            _ArrayItem(np.array(1, dtype=np.uint8), np.uint8),
            # Float values casted
            _ArrayItem(np.array(1, dtype=np.float16), np.int32, iscast=True),
        ],
    ),
    # `BoolArray` -> `bool_`
    # _DTypeTestItem(
    #     name='bool',
    #     cls_name='BoolArray',
    # ),
    # `StrArray` behavior is undefined for now (and should likely be tested
    # separately)
    # _DTypeTestItem(
    #     name='object',
    #     cls_name='StrArray',
    # ),
    _DTypeTestItem(
        name='bool',
        cls_name='bool_',
        array_items=[
            _ArrayItem(True, np.bool_),
            _ArrayItem(1, np.bool_, iscast=False),
            _ArrayItem([1.0], np.bool_, iscast=False),
            _ArrayItem(np.array([True], dtype=np.bool_), np.bool_),
            _ArrayItem(np.array([1], dtype=np.uint8), np.bool_, iscast=True),
        ],
    ),
    _DTypeTestItem(
        name='uint8',
        cls_name='ui8',
        array_items=[
            _ArrayItem(True, np.uint8, iscast=True),
            _ArrayItem(1, np.uint8),
            _ArrayItem([1.0], np.uint8, iscast=False),
            _ArrayItem(np.array([True], dtype=np.bool_), np.uint8, iscast=True),
            _ArrayItem(np.array([1], dtype=np.int32), np.uint8, iscast=True),
        ],
    ),
    _DTypeTestItem(
        name='uint16',
        cls_name='ui16',
    ),
    _DTypeTestItem(
        name='uint32',
        cls_name='ui32',
    ),
    _DTypeTestItem(
        name='uint64',
        cls_name='ui64',
    ),
    _DTypeTestItem(
        name='int8',
        cls_name='i8',
    ),
    _DTypeTestItem(
        name='int16',
        cls_name='i16',
    ),
    _DTypeTestItem(
        name='int32',
        cls_name='i32',
    ),
    _DTypeTestItem(
        name='int64',
        cls_name='i64',
    ),
    _DTypeTestItem(
        name='float16',
        cls_name='f16',
    ),
    _DTypeTestItem(
        name='float32',
        cls_name='f32',
    ),
    _DTypeTestItem(
        name='float64',
        cls_name='f64',
    ),
    _DTypeTestItem(
        name='complex64',
        cls_name='complex64',
    ),
    _DTypeTestItem(
        name='complex128',
        cls_name='complex128',
    ),
]


@pytest.mark.parametrize('item', _ALL_ITEMS)
@enp.testing.parametrize_xnp()
@pytest.mark.usefixtures('with_subtests')
def test_dtype(xnp, item: _DTypeTestItem):
  array_type = getattr(array_types, item.cls_name)
  dtype = array_type.dtype
  assert dtype.name == item.name
  assert dtype.array_cls_name == item.cls_name

  # DType casting
  for array_item in item.array_items:
    name = f'{array_item.value}:{array_item.dtype!r}'  # Name
    with epy.testing.subtest(name):
      assert_asarray(dtype=dtype, array_item=array_item, xnp=xnp)
    with epy.testing.subtest(f'{name}_nocast'):
      asarray_kwargs = dict(xnp=xnp, casting='none')
      if array_item.iscast:
        with pytest.raises(ValueError, match='Cannot cast'):
          dtype.asarray(array_item.value, **asarray_kwargs)
      else:
        assert_asarray(
            dtype=dtype,
            array_item=array_item,
            **asarray_kwargs,
        )


def assert_asarray(*, dtype, array_item: _ArrayItem, xnp, **asarray_kwargs):
  array = dtype.asarray(array_item.value, xnp=xnp, **asarray_kwargs)
  assert isinstance(array, xnp.ndarray)
  assert enp.lazy.as_dtype(array.dtype) == array_item.dtype
