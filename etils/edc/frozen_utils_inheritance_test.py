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

"""Dependency-light tests for inherited unfrozen methods."""

import dataclasses

from etils import edc
from etils.edc import frozen_utils
import pytest


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
