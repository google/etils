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

"""Tests for cast_utils."""

from __future__ import annotations

import dataclasses

from etils import edc
import pytest


@pytest.mark.parametrize('frozen', [True, False])
def test_edc(frozen: bool):
  class A:
    a: int

  class B(A):
    b_ignored: edc.AutoCast[int]  # Not a dataclass: ignored

  @edc.dataclass
  @dataclasses.dataclass(frozen=frozen)
  class C(B):
    c: edc.AutoCast[int]
    c_non_autocast: int

  @edc.dataclass
  @dataclasses.dataclass(frozen=frozen)
  class D(C):
    d: edc.AutoCast[
        int
    ] = '888'  # Default arg  # pytype: disable=annotation-type-mismatch

  d = D(c='123', c_non_autocast='456')  # pytype: disable=wrong-arg-types

  assert d.d == 888
  assert d.c == 123
  assert d.c_non_autocast == '456'

  if not frozen:
    d.d = '444'
    assert d.d == 444


@pytest.mark.parametrize('frozen', [True, False])
def test_cast_invalid_field(frozen: bool):

  @edc.dataclass
  @dataclasses.dataclass(frozen=frozen)
  class A:
    a: edc.AutoCast[int] = dataclasses.field(default_factory=int)

  with pytest.raises(ValueError, match='cannot be both `AutoCast`'):
    A()
