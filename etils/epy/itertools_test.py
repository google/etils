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

"""Tests for itertools."""

from __future__ import annotations

from etils import epy
import pytest


def test_group_by():
  out = epy.groupby(
      [0, 30, 2, 4, 2, 20, 3],
      key=lambda x: x < 10,
  )
  # Order is consistent with above
  assert out == {
      True: [0, 2, 4, 2, 3],
      False: [30, 20],
  }


def test_group_by_value():
  out = epy.groupby(
      ['111', '1', '11', '11', '4', '555'],
      key=len,
      value=int,
  )
  # Order is consistent with above
  assert out == {
      1: [1, 4],
      2: [11, 11],
      3: [111, 555],
  }


def test_zip_dict():
  d0 = {'a': 1, 'b': 2}
  d1 = {'a': 10, 'b': 20}
  assert list(epy.zip_dict(d0, d1)) == [
      ('a', (1, 10)),
      ('b', (2, 20)),
  ]

  # Order is preserved
  d0 = {'b': 1, 'a': 2}
  d1 = {'b': 10, 'a': 20}
  assert list(epy.zip_dict(d0, d1)) == [
      ('b', (1, 10)),
      ('a', (2, 20)),
  ]

  d0 = {'a': 1}
  d1 = {'a': 10, 'b': 20}
  with pytest.raises(KeyError):
    list(epy.zip_dict(d0, d1))

  with pytest.raises(KeyError):
    list(epy.zip_dict(d1, d0))


def test_zip_dict_three():
  d0 = {'a': 1, 'b': 2}
  d1 = {'a': 10, 'b': 20}
  d2 = {'a': 100, 'b': 200}

  assert list(epy.zip_dict(d0, d1, d2)) == [
      ('a', (1, 10, 100)),
      ('b', (2, 20, 200)),
  ]

  d2 = {'a': 100, 'b': 200, 'c': 300}
  with pytest.raises(KeyError):
    list(epy.zip_dict(d0, d1, d2))

  d2 = {'a': 100, 'c': 300}
  with pytest.raises(KeyError):
    list(epy.zip_dict(d0, d1, d2))


def test_issubclass():
  assert not epy.issubclass(1, int)
  assert epy.issubclass(bool, int)
