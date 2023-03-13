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

"""Tests."""

from __future__ import annotations

import datetime
import functools

from etils.ecolab.inspects import nodes
import numpy as np
import pytest


# TODO(epot): Could merge definition with `attrs_test.py`
class A:
  CONST = 'const'

  @property
  def x(self):
    return 123

  @property
  def y(self):
    raise ValueError

  def __getattr__(self, name):
    if name == 'dynamic_attr':
      return 'success'
    else:
      return super().__getattribute__(name)

  def __dir__(self):
    return ['aaa', 'dynamic_attr']


@pytest.mark.parametrize(
    'obj, cls',
    [
        (1, nodes.BuiltinNode),
        (None, nodes.BuiltinNode),
        ('aaa\nbbbb', nodes.BuiltinNode),
        (np.zeros, nodes.FnNode),
        (dict.keys, nodes.FnNode),
        (({}).keys, nodes.FnNode),
        (functools.partial(dict.keys), nodes.FnNode),
        (dict.keys, nodes.FnNode),
        (dict.__dict__['fromkeys'], nodes.FnNode),
        (datetime.timedelta.days, nodes.FnNode),
        (datetime.datetime.ctime.__call__, nodes.FnNode),
        (datetime.datetime.now().ctime.__call__, nodes.FnNode),
        ({'a': 123, True: 456}, nodes.MappingNode),
        ({'a', True, (3,)}, nodes.SetNode),
        ([{}, 123], nodes.SequenceNode),
        (np.zeros((2,)), nodes.ArrayNode),
        (dict, nodes.ClsNode),
        (A, nodes.ClsNode),
        (A(), nodes.ObjectNode),
    ],
)
def test_obj(obj: object, cls: type[nodes.Node]):
  node = nodes.Node.from_obj(obj)
  assert type(node) is cls  # pylint: disable=unidiomatic-typecheck
  assert isinstance(node.header_html, str)
  assert isinstance(node.inner_html, str)
