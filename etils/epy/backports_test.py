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

"""Tests for backports."""

import dataclasses

from etils import epy


def test_cached_property():

  @dataclasses.dataclass
  class A:
    x: int
    counter: int = 0

    @epy.cached_property
    def y(self):
      self.counter += 1
      return self.x * 10

  a = A(x=1)
  assert a.counter == 0
  assert a.y == 10  # pylint: disable=comparison-with-callable
  assert a.y == 10  # pylint: disable=comparison-with-callable
  a.x = 2  # Even after modifying x, y is still cached.
  assert a.y == 10  # pylint: disable=comparison-with-callable
  assert a.counter == 1
