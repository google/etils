# Copyright 2021 The etils Authors.
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

"""Python utils."""

import enum


class StrEnum(str, enum.Enum):
  """Like `Enum`, but `enum.auto()` assigns `str` rather than `int`.

  ```python
  class MyEnum(epy.StrEnum):
    SOME_ATTR = enum.auto()
    OTHER_ATTR = enum.auto()

  assert MyEnum('some_attr') is MyEnum.SOME_ATTR
  assert MyEnum.SOME_ATTR == 'some_attr'
  ```

  """

  def _generate_next_value_(name, start, count, last_values):  # pylint: disable=no-self-argument
    return name.lower()
