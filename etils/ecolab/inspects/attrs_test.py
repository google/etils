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

"""Test."""

from __future__ import annotations

from etils.ecolab.inspects import attrs as attrs_lib


class A:
  CONST = 'const'

  @property
  def x(self):
    return 123

  @property
  def y(self):
    raise ValueError

  def __getattr__(self, name):
    if name == 'dynamic_attr':
      return 'success'
    else:
      return super().__getattribute__(name)

  def __dir__(self):
    return ['aaa', 'dynamic_attr']


def test_attr():
  attrs = attrs_lib.get_attrs(A())
  # Magic methods extracted
  assert attrs['__class__'] == A

  # Attribute extracted, including descriptor / properties
  assert attrs['x'] == 123
  assert attrs['CONST'] == 'const'

  # Fault tolerant
  assert isinstance(attrs['y'], attrs_lib.ExceptionWrapper)

  # Extract dynamic `dir()` attributes
  assert attrs['dynamic_attr'] == 'success'
  assert isinstance(attrs['aaa'], attrs_lib.ExceptionWrapper)
