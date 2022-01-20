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

"""Tests for unfrozen."""

import dataclasses
from typing import Any

import chex
from etils import edc
import pytest


@edc.dataclass(allow_unfrozen=True)
@dataclasses.dataclass(frozen=True)
class A:
  x: Any = None
  y: Any = None

  def not_a_dataclass_attr(self):
    return 123


# Only the top-level dataclass has to be decorated
@dataclasses.dataclass(frozen=True)
class B:
  x: Any = None
  y: Any = None


# Inheritance supported
@dataclasses.dataclass(frozen=True)
class C(A):
  pass


# Test with chex dataclass
@edc.dataclass(allow_unfrozen=True)
@chex.dataclass(frozen=True)
class E:
  x: Any = None


def test_unfrozen_call_twice():
  x_origin = A(y=A(x=456))

  # Can't call frozen on frozen object
  with pytest.raises(ValueError, match='can only be called after'):
    x_origin.frozen()  # pytype: disable=attribute-error

  # Can call unfrozen twice on the original object
  x = x_origin.unfrozen()  # pytype: disable=attribute-error
  x = x_origin.unfrozen()  # pytype: disable=attribute-error

  # Can't call unfrozen on unfrozen objects
  with pytest.raises(ValueError, match='Object is already unfrozen'):
    x.unfrozen()  # pytype: disable=attribute-error

  y = x.y
  x.x = 123

  # Attribute still accessible
  assert x.not_a_dataclass_attr() == 123

  # Cannot set non-existing attributes
  with pytest.raises(AttributeError, match='object has no attribute'):
    _ = x.z

  with pytest.raises(AttributeError, match='Not a dataclass attribute'):
    x.not_a_dataclass_attr = 456

  # Only top level can be freezed
  with pytest.raises(ValueError, match='Only the top-level'):
    y.frozen()

  x_freezed = x.frozen()

  # After frozen has been called, cannot use the unfrozen objects anymore
  with pytest.raises(AttributeError, match='mutable was frozen'):
    _ = x.x
  with pytest.raises(AttributeError, match='mutable was frozen'):
    x.x = 123
  with pytest.raises(AttributeError, match='mutable was frozen'):
    _ = y.x
  with pytest.raises(AttributeError, match='mutable was frozen'):
    y.x = 123

  # Cannot call unfrozen twice on the original object
  x = x_origin.unfrozen()  # pytype: disable=attribute-error

  assert x_freezed == A(x=123, y=A(x=456))
  assert x_origin == A(y=A(x=456))


def test_unfrozen_original_obj_non_mutated():
  x_origin = A(x='abc', y=A(y='def'))

  x = x_origin.unfrozen()  # pytype: disable=attribute-error
  assert x.y.y == 'def'
  x.x = 123
  x.y = 456
  assert x.x == 123
  assert x.y == 456
  x = x.frozen()

  assert x == A(x=123, y=456)

  # x origin should be not mutated
  assert x_origin == A(x='abc', y=A(y='def'))


def test_unfrozen_assigned_twice():
  x_origin = A(x='abc', y=B(y='def'))

  x = x_origin.unfrozen()  # pytype: disable=attribute-error

  x.x = x.y  # Assign the `B()` to 2 different attributes
  x.y.x = 123  # Updating one update the other
  x.x.y = 456
  x = x.frozen()
  assert x == A(x=B(x=123, y=456), y=B(x=123, y=456))
  assert x_origin == A(x='abc', y=B(y='def'))


def test_unfrozen_nested():
  x_origin = A(x=B(x=A(x=123)))
  x = x_origin.unfrozen()  # pytype: disable=attribute-error
  val = A()
  x.x.x.x = A(x=val, y=A(y=123))
  x.x.x.x.y.y = 456

  x = x.frozen()
  assert x.x.x.x.x is val
  assert x == A(x=B(x=A(x=A(x=A(), y=A(y=456)))))
  assert x_origin == A(x=B(x=A(x=123)))


def test_unfrozen_inheritance():
  x_origin = C(x=B(x=A(x=123)))
  x = x_origin.unfrozen()  # pytype: disable=attribute-error
  val = A()
  x.x.x.x = A(x=val, y=A(y=123))
  x.x.x.x.y.y = 456

  x = x.frozen()
  assert isinstance(x, C)
  assert x.x.x.x.x is val
  assert x == C(x=B(x=A(x=A(x=A(), y=A(y=456)))))
  assert x_origin == C(x=B(x=A(x=123)))


def test_unfrozen_same_object():
  a = A(x=1, y=2)
  x_origin = A(x=a, y=a)

  x = x_origin.unfrozen()  # pytype: disable=attribute-error
  x.x.x = 10
  x = x.frozen()

  # Limitation: Even if `a.x` was modified, only the member which was accessed
  # was modified (x_origin.x but not `x_origin.y`), even if they where the
  # same instance before unfrozen.
  assert x == A(x=A(x=10, y=2), y=A(x=1, y=2))
  assert x_origin == A(x=A(x=1, y=2), y=A(x=1, y=2))

  x = x_origin.unfrozen()  # pytype: disable=attribute-error
  x.x.x = 10
  x.y.y = 20
  x = x.frozen()
  # Here, the 2 members are tracked, so we can correctly update them
  assert x.x is x.y
  assert x == A(x=A(x=10, y=20), y=A(x=10, y=20))
  assert x_origin == A(x=A(x=1, y=2), y=A(x=1, y=2))


def test_unfrozen_chex():
  x_origin = E(x=E(x=A(y=E(x=123))))
  x = x_origin.unfrozen()  # pytype: disable=attribute-error
  x.x.x.y.x = 456
  x = x.frozen()
  assert isinstance(x, E)
  assert x.x.x.y.x == 456
  assert x == E(x=E(x=A(y=E(x=456))))
  assert x_origin == E(x=E(x=A(y=E(x=123))))


def test_unfrozen_no_updates():
  x_origin = A(x=A(x=A(x=123)), y=A(y=A(y=456)))

  x = x_origin.unfrozen()  # pytype: disable=attribute-error
  assert x.x.x.x == 123  # Access read-only does not trigger `.replace`
  assert x.y.y.y == 456
  x = x.frozen()

  assert x_origin.x.x.x is x.x.x.x
  assert x_origin.x.x is x.x.x
  assert x_origin.x is x.x
  assert x_origin is x
  assert x_origin.y.y.y is x.y.y.y
  assert x_origin.y.y is x.y.y
  assert x_origin.y is x.y

  x = x_origin.unfrozen()  # pytype: disable=attribute-error
  x.x.x.x = 678
  assert x.y.y.y == 456
  x = x.frozen()

  # x is modifed, but not y
  assert x_origin.x.x.x is not x.x.x.x
  assert x_origin.x.x is not x.x
  assert x_origin is not x
  assert x_origin.y.y.y is x.y.y.y
  assert x_origin.y.y is x.y.y
  assert x_origin.y is x.y


# TODO(epot): Support circles
@pytest.mark.xfail
def test_unfrozen_cicle():
  x_origin = A(y=B(y=123))
  x = x_origin.unfrozen()  # pytype: disable=attribute-error
  x.y.y = x
  x = x.frozen()
  assert x.y.y is x
