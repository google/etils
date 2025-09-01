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

"""Descriptor utils."""

import typing
from typing import Any, Callable

_T = typing.TypeVar('_T')


if typing.TYPE_CHECKING:
  # TODO(b/171883689): There is likely a better way to annotate descriptors

  def classproperty(fn: Callable[[Any], _T]) -> _T:  # pylint: disable=function-redefined
    return fn(type(None))

else:

  class classproperty:  # pylint: disable=invalid-name
    """Decorator combining `property` and `classmethod`."""

    def __init__(self, fget):
      self.fget = fget

    def __get__(self, obj, cls):
      return self.fget(cls)
