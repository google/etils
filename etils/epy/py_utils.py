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

"""Python utils."""

from __future__ import annotations

import enum
import typing
from typing import Any, Union


class StrEnum(str, enum.Enum):
  """Like `Enum`, but `enum.auto()` assigns `str` rather than `int`.

  ```python
  class MyEnum(epy.StrEnum):
    SOME_ATTR = enum.auto()
    OTHER_ATTR = enum.auto()

  assert MyEnum('some_attr') is MyEnum.SOME_ATTR
  assert MyEnum.SOME_ATTR == 'some_attr'
  ```

  `StrEnum` is case insensitive.

  """

  # `issubclass(StrEnum, str)`, so can annotate `str` instead of `str | StrEnum`

  def _generate_next_value_(name, start, count, last_values) -> str:  # pylint: disable=no-self-argument
    return name.lower()

  @classmethod
  def _missing_(cls, value: str) -> StrEnum:
    if isinstance(value, str) and not value.islower():
      return cls(value.lower())
    # Could also add `did you meant yy ?`
    all_values = [e.value for e in cls]
    raise ValueError(f'{value!r} is not a valid {cls.__qualname__}. '
                     f'Expected one of {all_values}')

  def __eq__(self, other: str) -> bool:
    return super().__eq__(other.lower())

  def __hash__(self) -> int:  # pylint: disable=useless-super-delegation
    # Somehow `hash` is not defined automatically (maybe because of
    # the `__eq__`, so define it explicitly.
    return super().__hash__()

  # Pytype is confused by EnumMeta.__iter__ vs str.__iter__
  if typing.TYPE_CHECKING:

    @classmethod
    def __iter__(cls):
      return type(enum.Enum).__iter__(cls)


def issubclass_(
    cls: Any,
    types: Union[type[Any], tuple[type[Any], ...]],
) -> bool:
  """Like `issubclass`, but do not raise error if value is not `type`."""
  return isinstance(cls, type) and issubclass(cls, types)
