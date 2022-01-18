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

"""Dataclass utils."""

from __future__ import annotations

import functools
import typing
from typing import Callable, TypeVar

from etils.edc import frozen_utils

_Cls = TypeVar('_Cls')


@typing.overload
def dataclass(
    cls: None = ...,
    *,
    allow_unfrozen: bool = ...,
) -> Callable[[_Cls], _Cls]:
  ...


@typing.overload
def dataclass(
    cls: _Cls,
    *,
    allow_unfrozen: bool = ...,
) -> _Cls:
  ...


def dataclass(
    cls=None,
    *,
    allow_unfrozen=False,
):
  """Augment a dataclass with additional features.

  `allow_unfrozen`: allow nested dataclass to be updated. This add two methods:

   * `.unfrozen()`: Create a lazy deep-copy of the current dataclass. Updates
     to nested attributes will be propagated to the top-level dataclass.
   * `.frozen()`: Returns the frozen dataclass, after it was mutated.

  Example:

  ```python
  old_x = X(y=Y(z=123))

  x = old_x.unfrozen()
  x.y.z = 456
  x = x.frozen()

  assert x == X(y=Y(z=123))  # Only new x is mutated
  assert old_x == X(y=Y(z=456))  # Old x is not mutated
  ```

  Note:

  * Only the last `.frozen()` call resolve the dataclass by calling `.replace`
    recursivelly.
  * Dataclass returned by `.unfrozen()` and nested attributes are not the
    original dataclass but proxy objects which track the mutations. As such,
    those object are not compatible with `isinstance()`, `jax.tree_map`,...
  * Only the top-level dataclass need to be `allow_unfrozen=True`
  * After `.frozen()` has been called, any of the temporary sub-attribute
    become invalid:

    ```python
    a = a.unfrozen()
    y = a.y
    a = a.frozen()

    y.x  # Raise error (created between the unfrozen/frozen call)
    a.y.x  # Work
    ```

  Args:
    cls: The dataclass to decorate
    allow_unfrozen: If True, add `.frozen`, `.unfrozen` methods.

  Returns:
    Decorated class
  """
  # Return decorator
  if cls is None:
    return functools.partial(
        dataclass,
        allow_unfrozen=allow_unfrozen,
    )

  if allow_unfrozen:
    cls = frozen_utils.add_unfrozen(cls)

  return cls
