# Copyright 2025 The etils Authors.
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

"""Context stack utils."""

from __future__ import annotations

import dataclasses
import typing
from typing import Generic, TypeVar

from etils.edc import context
from etils.edc import dataclass_utils

_T = TypeVar('_T')


# TODO(epot): Should likely be exposed in `epy.ContextStack` rather than `edc`
# TODO(epot): The `ContextStack` object should likely implement the `Sequence`
# interface, so it can be used directly ?


@dataclass_utils.dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class ContextStack(Generic[_T]):
  """Local stack object (per-thread and contextvars-aware).

  Each thread / coroutine will have its own stack. This ensure that
  contextmanagers from different threads do not conflict.

  Usage:

  ```python
  _stack = edc.ContextStack[int]()

  @contextlib.contextmanager
  def my_thread_safe_contextmanager():
    _stack.append(1)
    try:
      yield
    finally:
      _stack.pop()
  ```
  """

  if typing.TYPE_CHECKING:
    stack: list[_T] = dataclasses.field(default_factory=list)
  else:
    stack: context.ContextVar[list[_T]] = dataclasses.field(
        default_factory=list
    )

  def append(self, value: _T) -> None:
    """Append a value to the stack."""
    self.stack.append(value)

  def pop(self) -> _T:
    """Pop the last value from the stack."""
    return self.stack.pop()

  def __getitem__(self, index: int) -> _T:
    """Get the item at the given index."""
    return self.stack[index]

  def __len__(self) -> int:
    """Get the length of the stack."""
    return len(self.stack)
