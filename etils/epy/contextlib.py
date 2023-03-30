# Copyright 2023 The etils Authors.
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

"""Contextmanager utils."""

from __future__ import annotations

import abc
import contextlib
from typing import Generic, Iterable, TypeVar

_T = TypeVar('_T')

# TODO(epot): Support
# * Per-instance (done)
# * Inheritance with `super()`
# * Multi-thread
# * Covariable
# * Re-entry
# * Immutable classes


class ContextManager(abc.ABC, Generic[_T]):
  """ContextManager allows to define contextmanager class using yield-syntax.

  Example:

  ```python
  class A(epy.ContextManager):

    def __contextmanager__(self) -> Iterable[A]:
      yield self


  with A() as a:
    pass
  ```

  One the code is more mature, this could be merged to `contextlib` directly.
  https://discuss.python.org/t/yield-based-contextmanager-for-classes/8453

  """

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)

    # Check whether the class is already wrapped in a CM
    if not hasattr(cls.__contextmanager__, '_cm_added'):
      cls.__contextmanager__ = contextlib.contextmanager(cls.__contextmanager__)
      cls.__contextmanager__._cm_added = (
          True  # pylint: disable=protected-access
      )

  @abc.abstractmethod
  def __contextmanager__(self) -> Iterable[_T]:
    pass

  __contextmanager__._cm_added = True  # pylint: disable=protected-access

  def __enter__(self) -> _T:
    self._epy_cm = self.__contextmanager__()
    return self._epy_cm.__enter__()  # pytype: disable=attribute-error

  def __exit__(self, exc_type, exc_value, traceback) -> None:
    return self._epy_cm.__exit__(exc_type, exc_value, traceback)  # pytype: disable=attribute-error
