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

"""Test."""

import collections
from etils import etree


def test_py_map():
  tree0 = {
      'b': 1,
      'a': [
          (),
          collections.UserDict(a=5),
          [{3: 2, 1: 4}],
      ],
  }
  tree1 = {
      'b': 10,
      'a': [
          (),
          collections.UserDict(a=10),
          [{3: 10, 1: 20}],
      ],
  }
  expected_tree = {
      'b': 11,
      'a': [
          (),
          collections.UserDict(a=15),
          [{3: 12, 1: 24}],
      ],
  }
  tree_out = etree.py.map(lambda x, y: x + y, tree0, tree1)
  assert tree_out == expected_tree
  # type preserved
  assert isinstance(tree_out['a'][1], collections.UserDict)
  # dict key order preserved
  assert list(tree_out) == ['b', 'a']

  # Round-trip flatten / unflatten
  flat0, tree_struct = etree.py.backend.flatten(tree0)

  assert flat0 == [5, 4, 2, 1]
  assert etree.py.backend.unflatten(tree_struct, flat0) == tree_struct
