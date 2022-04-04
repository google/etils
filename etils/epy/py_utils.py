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
import itertools
from typing import Any, Iterator, TypeVar

# from typing_extensions import Unpack, TypeVarTuple  # pytype: disable=not-supported-yet  # pylint: disable=g-multiple-import

# TODO(pytype): Once supported, should replace
Unpack = Any
TypeVarTuple = Any

_KeyT = TypeVar('_KeyT')
_ValuesT = Any  # TypeVarTuple('_ValuesT')


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
    if isinstance(value, str):
      return cls(value.lower())
    return super()._missing_(value)

  def __eq__(self, other: str) -> bool:
    return super().__eq__(other.lower())

  def __hash__(self) -> int:  # pylint: disable=useless-super-delegation
    # Somehow `hash` is not defined automatically (maybe because of
    # the `__eq__`, so define it explicitly.
    return super().__hash__()


# pyformat: disable
def zip_dict(  # pytype: disable=invalid-annotation
    *dicts: Unpack[dict[_KeyT, _ValuesT]],
) -> Iterator[_KeyT, tuple[Unpack[_ValuesT]]]:
  """Iterate over items of dictionaries grouped by their keys.

  Example:

  ```python
  d0 = {'a': 1, 'b': 2}
  d1 = {'a': 10, 'b': 20}
  d2 = {'a': 100, 'b': 200}

  list(epy.zip_dict(d0, d1, d2)) == [
      ('a', (1, 10, 100)),
      ('b', (2, 20, 200)),
  ]
  ```

  Args:
    *dicts: The dict to iterate over. Should all have the same keys

  Yields:
    The iterator of `(key, zip(*values))`

  Raises:
    KeyError: If dicts does not contain the same keys.
  """
  # pyformat: enable
  # Set does not keep order like dict, so only use set to compare keys
  all_keys = set(itertools.chain(*dicts))
  d0 = dicts[0]

  if len(all_keys) != len(d0):
    raise KeyError(f'Missing keys: {all_keys ^ set(d0)}')

  for key in d0:  # set merge all keys
    # Will raise KeyError if the dict don't have the same keys
    yield key, tuple(d[key] for d in dicts)
