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

"""Tests for dataclass_utils."""

import dataclasses
from typing import Any

from etils import edc
from etils import epy
import pytest


def test_kw_only():

  @edc.dataclass(kw_only=True)
  @dataclasses.dataclass(frozen=True)
  class KwOnly:
    x: Any
    y: Any

  a = KwOnly(x=1, y=2)
  assert a.x == 1
  assert a.y == 2

  with pytest.raises(TypeError, match='contructor is keyword-only.'):
    _ = KwOnly(1, 2)


@edc.dataclass
@dataclasses.dataclass(frozen=True)
class A:
  x: Any = None
  y: Any = None


@edc.dataclass
@dataclasses.dataclass(frozen=True)
class B:
  x: Any = None
  y: Any = None

  def replace(self) -> int:  # Custom replace function
    return 123


def test_replace():
  obj = object()
  x = A(y=obj)
  y = x.replace(x=123)  # pytype: disable=attribute-error
  assert x == A(y=obj)
  assert y == A(x=123, y=obj)
  assert x.y is y.y

  assert B().replace() == 123


@edc.dataclass
@dataclasses.dataclass
class R:
  x: Any = None
  y: Any = None


@edc.dataclass
@dataclasses.dataclass
class R1(R):
  z: Any = None


class R11(R1):  # Is not dataclass but `__name__` should be correct
  pass


@edc.dataclass
@dataclasses.dataclass
class R2(R):
  z: Any = None

  def __repr__(self):
    return 'R2 repr'


@edc.dataclass
@dataclasses.dataclass
class R0Field():
  pass


@edc.dataclass
@dataclasses.dataclass
class R1Field():
  x: Any = None


def test_repr():
  assert repr(R(123, R11(y='abc'))) == epy.dedent("""
  R(
      x=123,
      y=R11(
          x=None,
          y='abc',
          z=None,
      ),
  )
  """)

  # Curstom __repr__
  assert repr(R2()) == 'R2 repr'

  # When 1 or 0 field, print in a single line
  assert repr(R0Field()) == 'R0Field()'
  assert repr(R1Field()) == 'R1Field(x=None)'

  # Recursive
  x = R()
  x.x = x
  assert repr(x) == epy.dedent("""
  R(
      x=...,
      y=None,
  )
  """)
