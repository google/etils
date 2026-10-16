# Copyright 2026 The etils Authors.
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
from etils import epy
from etils.edc import frozen_utils
import jax
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

  assert repr(x) == epy.dedent("""
      _MutableProxy(A(
          x=123,
          y=A(x=456, y=None),
      ))""")

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


def test_unfrozen_assigned_object():
  x_origin = A(x=A(x=123))

  x = x_origin.unfrozen()  # pytype: disable=attribute-error

  x.x = A(x=456)
  x.x.y = 567  # pyrefly: ignore[read-only]
  x = x.frozen()
  assert x == A(x=A(x=456, y=567))
  assert x_origin == A(x=A(x=123))


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


# Could try to optimize read access. Currently, any `getattr` will trigger
# a replace, even if the attribute was not updated.
@pytest.mark.xfail
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


def test_unfrozen_tree_map():
  x_origin = E(x=E(x=456))
  x = x_origin.unfrozen()  # pytype: disable=attribute-error

  y = jax.tree_util.tree_map(lambda v: v * 10, x.x)
  assert isinstance(y, E)
  assert y == E(x=4560)
  x.x = y

  x = x.frozen()

  # x origin should be not mutated
  assert x_origin == E(x=E(x=456))
  assert x == E(x=E(x=4560))


# TODO(epot): Support circles
@pytest.mark.xfail
def test_unfrozen_cicle():
  x_origin = A(y=B(y=123))
  x = x_origin.unfrozen()  # pytype: disable=attribute-error
  x.y.y = x
  x = x.frozen()
  assert x.y.y is x


@pytest.mark.parametrize('use_factory', [False, True])
@pytest.mark.parametrize('slots', [False, True])
@pytest.mark.parametrize('redecorate_middle', [False, True])
def test_unfrozen_redecorated_inheritance(use_factory, slots, redecorate_middle):
  def decorate(cls):
    if use_factory:
      return edc.dataclass(allow_unfrozen=True)(cls)
    return edc.dataclass(cls, allow_unfrozen=True)

  @decorate
  @dataclasses.dataclass(frozen=True, slots=slots)
  class Parent:
    x: int = 1

  @dataclasses.dataclass(frozen=True, slots=slots)
  class Middle(Parent):
    y: int = 2

  if redecorate_middle:
    Middle = decorate(Middle)

  @decorate
  @dataclasses.dataclass(frozen=True, slots=slots)
  class Child(Middle):
    z: int = 3
    nested: Parent = dataclasses.field(default_factory=Parent)

  # Redecoration must preserve the inherited methods, not shadow them.
  assert Child.frozen is Parent.frozen is frozen_utils.frozen
  assert Child.unfrozen is Parent.unfrozen is frozen_utils.unfrozen
  assert 'unfrozen' not in Child.__dict__
  original = Child()
  with pytest.raises(dataclasses.FrozenInstanceError):
    original.z = 10
  proxy = original.unfrozen()
  nested = proxy.nested
  proxy.x = 10
  proxy.y = 20
  proxy.z = 30
  nested.x = 40
  result = proxy.frozen()
  assert type(result) is Child
  assert result == Child(x=10, y=20, z=30, nested=Parent(x=40))
  assert original == Child()
  with pytest.raises(AttributeError, match='after the mutable was frozen'):
    _ = nested.x
  with pytest.raises(AttributeError, match='after the mutable was frozen'):
    proxy.z = 50
  with pytest.raises(ValueError, match='can only be called after'):
    original.frozen()
  assert original.unfrozen().frozen() is original


@pytest.mark.parametrize('name', ['frozen', 'unfrozen'])
@pytest.mark.parametrize('inherited', [False, True])
@pytest.mark.parametrize('use_factory', [False, True])
def test_unfrozen_custom_method_collision(name, inherited, use_factory):
  def custom(self):
    return 'custom'

  @dataclasses.dataclass(frozen=True)
  class Parent:
    x: int = 1

  if inherited:
    setattr(Parent, name, custom)

  @dataclasses.dataclass(frozen=True)
  class Child(Parent):
    pass

  if not inherited:
    setattr(Child, name, custom)
  with pytest.raises(ValueError, match='already define `frozen` or `unfrozen`'):
    if use_factory:
      edc.dataclass(allow_unfrozen=True)(Child)
    else:
      edc.dataclass(Child, allow_unfrozen=True)
  assert getattr(Child, name) is custom


@pytest.mark.parametrize('name', ['frozen', 'unfrozen'])
def test_unfrozen_partial_implementation(name):
  @dataclasses.dataclass(frozen=True)
  class Partial:
    pass

  setattr(Partial, name, getattr(frozen_utils, name))
  with pytest.raises(ValueError, match='already define'):
    edc.dataclass(Partial, allow_unfrozen=True)


@pytest.mark.parametrize('use_factory', [False, True])
def test_unfrozen_unrelated_has_unfrozen(use_factory):
  @dataclasses.dataclass(frozen=True)
  class Data:
    x: int = 1

    def has_unfrozen(self):
      return 'unrelated'

  if use_factory:
    Data = edc.dataclass(allow_unfrozen=True)(Data)
  else:
    Data = edc.dataclass(Data, allow_unfrozen=True)
  assert Data().has_unfrozen() == 'unrelated'
  proxy = Data().unfrozen()
  proxy.x = 2
  assert proxy.frozen() == Data(x=2)


@pytest.mark.parametrize('flag', [None, False, True])
@pytest.mark.parametrize('use_factory', [False, True])
def test_unfrozen_flag_and_repeat_decoration(flag, use_factory):
  @dataclasses.dataclass(frozen=True)
  class Data:
    x: int = 1

  kwargs = {} if flag is None else {'allow_unfrozen': flag}
  for _ in range(2):
    if use_factory:
      Data = edc.dataclass(**kwargs)(Data)
    else:
      Data = edc.dataclass(Data, **kwargs)
  assert hasattr(Data, 'unfrozen') is (flag is True)
  if flag:
    assert Data().unfrozen().frozen() == Data()


@pytest.mark.parametrize('flag', [None, False])
def test_unfrozen_inherited_without_flag(flag):
  @edc.dataclass(allow_unfrozen=True)
  @dataclasses.dataclass(frozen=True)
  class Parent:
    x: int = 1

  @dataclasses.dataclass(frozen=True)
  class Child(Parent):
    y: int = 2

  kwargs = {} if flag is None else {'allow_unfrozen': flag}
  Child = edc.dataclass(Child, **kwargs)
  proxy = Child().unfrozen()
  proxy.y = 3
  assert proxy.frozen() == Child(y=3)


@pytest.mark.parametrize('name', ['frozen', 'unfrozen'])
@pytest.mark.parametrize('default', [None, False, 0, '', 1])
@pytest.mark.parametrize('use_factory', [False, True])
def test_unfrozen_default_field_collision(name, default, use_factory):
  # Use actual dataclass fields: their class defaults participate in getattr.
  Data = dataclasses.make_dataclass(
      'Data', [(name, object, dataclasses.field(default=default))], frozen=True
  )

  def decorate():
    if use_factory:
      return edc.dataclass(allow_unfrozen=True)(Data)
    return edc.dataclass(Data, allow_unfrozen=True)

  if default is not None:
    # Even falsey non-None defaults are collisions, not missing attributes.
    with pytest.raises(ValueError, match='already define `frozen` or `unfrozen`'):
      decorate()
    assert getattr(Data, name) is default
  else:
    # Preserve the existing None-as-missing rule. The generated class method
    # replaces None, but __init__ still sets the instance field to None.
    assert decorate() is Data
    assert getattr(Data, name) is getattr(frozen_utils, name)
    assert getattr(Data(), name) is None
  assert dataclasses.fields(Data)[0].default is default
