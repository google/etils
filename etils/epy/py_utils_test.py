# Copyright 2024 The etils Authors.
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

import collections
import enum
import sys
import typing

from etils import epy
import pytest


class MyEnum(epy.StrEnum):
  MY_OTHER_ATTR = enum.auto()
  MY_ATTR = enum.auto()
  MY_UPPER_ATTR = 'My_Upper_Attr'
  OTHER = 'not_Other'


def test_str_enum():
  if sys.version_info[:2] <= (3, 11):
    return  # Skip for 3.11

  assert MyEnum.MY_ATTR is MyEnum.MY_ATTR
  assert MyEnum('my_attr') is MyEnum.MY_ATTR
  assert MyEnum('MY_ATTR') is MyEnum.MY_ATTR
  assert MyEnum('MY_UPPER_ATTR') is MyEnum.MY_UPPER_ATTR
  assert MyEnum('my_upper_attr') is MyEnum.MY_UPPER_ATTR
  assert MyEnum.MY_ATTR == MyEnum.MY_ATTR
  assert MyEnum.MY_ATTR == 'my_attr'
  assert MyEnum.MY_ATTR == 'MY_ATTR'
  assert MyEnum.MY_UPPER_ATTR == MyEnum.MY_UPPER_ATTR
  assert MyEnum.MY_UPPER_ATTR == 'MY_UPPER_ATTR'
  assert MyEnum.MY_UPPER_ATTR == 'my_upper_attr'
  assert MyEnum.MY_UPPER_ATTR != MyEnum.MY_ATTR
  assert MyEnum.MY_UPPER_ATTR != 'my_attr'
  assert MyEnum.MY_UPPER_ATTR != 'unknonw_my_attr'
  assert MyEnum.OTHER == MyEnum.OTHER
  assert MyEnum.OTHER is MyEnum.OTHER
  assert MyEnum.OTHER == 'not_other'
  assert MyEnum.OTHER == 'NOT_OTHER'
  assert MyEnum.OTHER != MyEnum.MY_ATTR
  assert MyEnum.OTHER != 'my_attr'
  assert MyEnum.OTHER != 'unknonw_my_attr'

  assert hash(MyEnum.MY_ATTR) == hash('my_attr')

  with pytest.raises(ValueError, match='Expected one of'):
    MyEnum('non-existing')

  assert [e.value for e in MyEnum] == [
      'my_other_attr',
      'my_attr',
      'My_Upper_Attr',
      'not_Other',
  ]


def test_is_namedtuple():
  class MyTuple0(typing.NamedTuple):
    x: int
    y: int

  MyTuple1 = collections.namedtuple('MyTuple1', 'x, y')

  epy.is_namedtuple(MyTuple0(0, 1))
  epy.is_namedtuple(MyTuple1(0, 1))


@epy.frozen
class A:

  def __init__(self):
    self.x = 123
    self.y = 456


def test_frozen():
  a = A()
  assert a.x == 123
  with pytest.raises(AttributeError):
    a.x = 456

  with pytest.raises(AttributeError):
    a.w = 456


def test_frozen_inheritance_no_init():
  class B(A):
    pass

  a = B()
  assert a.x == 123
  with pytest.raises(AttributeError):
    a.x = 456

  with pytest.raises(AttributeError):
    a.w = 456


def test_frozen_inheritance_missing_frozen():
  class B(A):

    def __init__(self):
      # Before, raise an error
      with pytest.raises(ValueError, match='Child of `@epy.frozen`'):
        self.x = 456
      super().__init__()
      # After, default error is raised
      with pytest.raises(AttributeError):
        self.x = 456

  b = B()
  assert b.x == 123

  with pytest.raises(AttributeError):
    b.w = 456


def test_frozen_inheritance_new_init():
  @epy.frozen
  class B(A):

    def __init__(self):
      self.x2 = 123
      super().__init__()
      self.x3 = 123

  b = B()
  assert b.x == 123
  assert b.x2 == 123
  assert b.x3 == 123
  with pytest.raises(AttributeError):
    b.w = 456


def test_issubclass():
  assert not epy.issubclass(1, int)
  assert epy.issubclass(bool, int)


def test_equality():
  x = MyEnum.MY_ATTR
  assert x == 'MY_ATTR'
  assert x == 'my_attr'
  assert not x != 'MY_ATTR'  # pylint: disable=g-comparison-negation
  assert not x != 'my_attr'  # pylint: disable=g-comparison-negation

  assert x != 'MY_OTHER_ATTR'
  assert x != MyEnum.MY_OTHER_ATTR
