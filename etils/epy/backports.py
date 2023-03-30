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

"""Python 3.8+ backports."""

import typing
from typing import Any, Callable, TypeVar

_T = TypeVar('_T')


class cached_property(property):  # pylint: disable=invalid-name
  """Backport of py3.8 `functools.cached_property`."""

  def __get__(self, obj, objtype=None):
    # See https://docs.python.org/3/howto/descriptor.html#properties
    if obj is None:
      return self
    if self.fget is None:  # pytype: disable=attribute-error
      raise AttributeError('Unreadable attribute.')
    attr = '__cached_' + self.fget.__name__  # pytype: disable=attribute-error
    cached = getattr(obj, attr, None)
    if cached is None:
      cached = self.fget(obj)  # pytype: disable=attribute-error
      # Use `object.__setattr__` for compatibility with frozen dataclasses
      object.__setattr__(obj, attr, cached)
    return cached


if typing.TYPE_CHECKING:
  # TODO(b/171883689): There is likelly better way to annotate descriptors

  def cached_property(fn: Callable[[Any], _T]) -> _T:  # pylint: disable=function-redefined
    return fn(None)
