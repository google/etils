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

"""Tests for field_utils."""

from __future__ import annotations

import dataclasses
from typing import Any

from etils import edc
import pytest


@pytest.mark.parametrize('frozen', [True, False])
def test_field_no_op(frozen: bool):

  @dataclasses.dataclass(frozen=frozen)
  class A:
    x: Any = edc.field()  # pytype: disable=annotation-type-mismatch

  # No argument
  with pytest.raises(TypeError, match=r'__init__\(\) missing 1 required'):
    A()

  # Default argument
  a = A(123)
  assert a.x == 123

  if frozen:
    with pytest.raises(dataclasses.FrozenInstanceError):
      a.x = 456
  else:
    a.x = 456
    assert a.x == 456


@pytest.mark.parametrize('frozen', [True, False])
def test_field_multi_instance(frozen: bool):
  """Each instance has a separate state."""

  @dataclasses.dataclass(frozen=frozen)
  class A:
    x: Any = edc.field(validate=str)  # pytype: disable=annotation-type-mismatch

  a0 = A(123)
  assert a0.x == '123'
  a1 = A(456)
  assert a0.x == '123'
  assert a1.x == '456'

  if frozen:
    with pytest.raises(dataclasses.FrozenInstanceError):
      a0.x = 456
  else:
    a0.x = 789
    assert a0.x == '789'


@pytest.mark.parametrize('repr_', [True, False])
@pytest.mark.parametrize('eq', [True, False])
@pytest.mark.parametrize('order', [True, False])
@pytest.mark.parametrize('unsafe_hash', [True, False])
@pytest.mark.parametrize('frozen', [True, False])
@pytest.mark.parametrize(
    'default_kwargs',
    [{}, dict(default=789),
     dict(default_factory=lambda: 789)],
)
def test_field_validate(
    repr_: bool,
    eq: bool,
    order: bool,
    unsafe_hash: bool,
    frozen: bool,
    default_kwargs: dict[str, Any],
):
  """Make sure that default factory works for all combinations of dataclass."""
  if order and not eq:  # eq must be true if order is true
    return

  @dataclasses.dataclass(
      # default_factory doesn't make sense if init = False
      init=True,
      repr=repr_,
      eq=eq,
      order=order,
      unsafe_hash=unsafe_hash,
      frozen=frozen,  # pytype: disable=not-supported-yet
  )
  class A:
    x: Any = edc.field(validate=str, **default_kwargs)  # pytype: disable=annotation-type-mismatch

  # No argument
  if default_kwargs:
    a = A()
    assert a.x == '789'
  else:
    # No argument
    with pytest.raises(TypeError, match=r'__init__\(\) missing 1 required'):
      A()

  # Single argument
  a = A(456)
  assert a.x == '456'

  if repr_:
    assert repr(a) == "test_field_validate.<locals>.A(x='456')"

  # Updating the value should only work for non-frozen instances.
  if frozen:
    with pytest.raises(dataclasses.FrozenInstanceError):
      a.x = 678
  else:
    a.x = 678
    assert a.x == '678'

  # Class attribute
  assert isinstance(A.x, edc.field_utils._Field)
