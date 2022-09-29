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

"""Tests for contextlib."""

from __future__ import annotations

import dataclasses
from typing import Iterable

from etils import epy


@dataclasses.dataclass
class A(epy.ContextManager):
  x: int
  state: list[str] = dataclasses.field(default_factory=list)

  def __contextmanager__(self) -> Iterable[A]:
    self.state.append(f'start:{self.x}')
    yield self
    self.state.append(f'end:{self.x}')


def test_contextmanager():
  with A(1) as a1:
    assert a1.x == 1

    # Different instances can be nested (and don't share state)
    with A(2) as a2:
      assert a2.x == 2

    assert a1.state == ['start:1']
    assert a2.state == ['start:2', 'end:2']
  assert a1.state == ['start:1', 'end:1']

  # CM can be re-openned (after being closed)
  with a1 as a11:
    assert a11.x == 1
    assert a1.state == ['start:1', 'end:1', 'start:1']

  assert a1.state == ['start:1', 'end:1', 'start:1', 'end:1']
