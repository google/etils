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

"""Tests for tree_utils."""

import dataclasses
import sys

from etils import enp
from etils import etree as etree_lib
import numpy as np
import pytest

# pylint: disable=g-bad-import-order,g-import-not-at-top
import jax.numpy as jnp
import tensorflow as tf
# pylint: enable=g-bad-import-order,g-import-not-at-top


@pytest.fixture(
    params=[  # Run all tests 3 times (one for each backends)
        etree_lib.jax,
        etree_lib.tree,
        etree_lib.nest,
    ],)
def etree_api(request):
  yield request.param


def test_tree_parallel_map(etree_api: etree_lib.tree_utils.TreeAPI):  # pylint: disable=redefined-outer-name
  assert etree_api.parallel_map(lambda x: x * 10, {
      'a': [1, 2, 3],
      'b': [4, 5]
  }) == {
      'a': [10, 20, 30],
      'b': [40, 50]
  }


def test_tree_parallel_map_reraise(etree_api: etree_lib.tree_utils.TreeAPI):  # pylint: disable=redefined-outer-name

  def fn(x):
    raise ValueError('Bad value')

  with pytest.raises(ValueError, match='Bad value'):
    etree_api.parallel_map(fn, [1])


def test_tree_unzip(etree_api):  # pylint: disable=redefined-outer-name
  unflatten = [{'a': 1, 'b': 10}, {'a': 2, 'b': 20}, {'a': 3, 'b': 30}]
  assert list(
      etree_api.unzip({
          'a': np.array([1, 2, 3]),
          'b': np.array([10, 20, 30]),
      })) == unflatten


@dataclasses.dataclass
class Obj:
  pass


def test_spec_like(etree_api: etree_lib.tree_utils.TreeAPI):  # pylint: disable=redefined-outer-name
  obj = Obj()

  values = [
      # tf
      tf.TensorSpec((None,), dtype=tf.int32),
      # jax
      jnp.zeros((6,), dtype=np.int32),
      # np
      {
          'a': np.ones((7,), dtype=np.float32),
      },
      enp.array_spec._get_none_spec(),
      np.array(['abc', 'def']),
      # Other values are pass-through
      None,
      123,
      'abc',
      obj,
  ]

  specs = etree_api.spec_like(values)
  assert specs == [
      enp.ArraySpec((None,), dtype=np.int32),
      enp.ArraySpec((6,), dtype=np.int32),
      {
          'a': enp.ArraySpec((7,), dtype=np.float32)
      },
      None,
      enp.ArraySpec((2,), dtype=str),
      None,
      123,
      'abc',
      obj,
  ]

  out = "[i32[_], i32[6], {'a': f32[7]}, None, str[2], None, 123, 'abc', Obj()]"
  assert repr(specs) == out


def test_tree_assert_same_structure(etree_api: etree_lib.tree_utils.TreeAPI):  # pylint: disable=redefined-outer-name
  etree_api.backend.assert_same_structure(
      {
          'x': [1, 2, 3],
          'y': [],
      },
      {
          'x': [10, 20, 30],
          'y': [],
      },
  )

  with pytest.raises(ValueError, match='The two structures don'):
    etree_api.backend.assert_same_structure(
        {
            'x': [1, 2, 3],
            'y': [],
        },
        {
            'x': [10, 20, 30],
            'y2': [],
        },
    )
