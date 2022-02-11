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

"""Dataclass utils."""

from __future__ import annotations

import dataclasses
import functools
import reprlib
import typing
from typing import Any, Callable, TypeVar

from etils import epy
from etils.edc import frozen_utils

_Cls = Any
_ClsT = TypeVar('_ClsT')
_T = TypeVar('_T')


@typing.overload
def dataclass(
    cls: None = ...,
    *,
    kw_only: bool = ...,
    replace: bool = ...,  # pylint: disable=redefined-outer-name
    repr: bool = ...,  # pylint: disable=redefined-builtin
    allow_unfrozen: bool = ...,
) -> Callable[[_ClsT], _ClsT]:
  ...


@typing.overload
def dataclass(
    cls: _ClsT,
    *,
    kw_only: bool = ...,
    replace: bool = ...,  # pylint: disable=redefined-outer-name
    repr: bool = ...,  # pylint: disable=redefined-builtin
    allow_unfrozen: bool = ...,
) -> _ClsT:
  ...


def dataclass(
    cls=None,
    *,
    kw_only=False,
    replace=True,  # pylint: disable=redefined-outer-name
    repr=True,  # pylint: disable=redefined-builtin
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
  * Avoid using `unfrozen` if 2 attributes of the dataclass point to the
    same nested dataclass. Updates on one attribute might not be reflected on
    the other.

    ```python
    y = Y(y=123)
    x = X(x0=y, x1=y)  # Same instance assigned twice in `x0` and `x1`
    x = x.unfrozen()
    x.x0.y = 456  # Changes in `x0` not reflected in `x1`
    x = x.frozen()

    assert x == X(x0=Y(y=456), x1=Y(y=123))
    ```

    This is because only attributes which are accessed are tracked, so `etils`
    do not know the object exist somewhere else in the attribute tree.

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
    kw_only: If True, make the dataclass `__init__` keyword-only.
    replace: If `True`, add a `.replace(` alias of `dataclasses.replace`.
    repr: If `True`, the class `__repr__` will return a pretty-printed `str`
      (one attribute per line)
    allow_unfrozen: If `True`, add `.frozen`, `.unfrozen` methods.

  Returns:
    Decorated class
  """
  # Return decorator
  if cls is None:
    return functools.partial(
        dataclass,
        kw_only=kw_only,
        replace=replace,
        repr=repr,
        allow_unfrozen=allow_unfrozen,
    )

  if kw_only:
    cls = _make_kw_only(cls)

  if repr:
    cls = add_repr(cls)

  if replace:
    cls = _add_replace(cls)

  if allow_unfrozen:
    cls = frozen_utils.add_unfrozen(cls)

  return cls


def _make_kw_only(cls: _ClsT) -> _ClsT:
  """Replace the `__init__` by a keyword-only version."""
  # Use `cls.__dict__` and not `hasattr` to ignore parent classes
  if '__init__' not in cls.__dict__:
    return cls  # Do not mutate the class if __init__ isn't present

  old_init = cls.__init__

  # Despite `@functools.wraps`, the function has to be called `__init__` (
  # see: https://stackoverflow.com/q/29919804/4172685)
  @functools.wraps(old_init)
  def __init__(self, *args, **kwargs):  # pylint: disable=invalid-name
    if args:
      raise TypeError(f'{self.__class__.__name__} contructor is keyword-only. '
                      f'Got {len(args)} positional arguments.')
    return old_init(self, **kwargs)

  cls.__init__ = __init__

  return cls


def _add_replace(cls: _ClsT) -> _ClsT:
  """Add a `.replace` method to the class, if not already present."""
  # Use `cls.__dict__` and not `hasattr` to ignore parent classes
  if 'replace' not in cls.__dict__:
    cls.replace = replace
  return cls


def replace(self: _T, **kwargs: Any) -> _T:
  """Similar to `dataclasses.replace`."""
  return dataclasses.replace(self, **kwargs)


def has_default_repr(cls: _Cls) -> bool:
  return (
      # Use `cls.__dict__` and not `hasattr` to ignore parent classes
      '__repr__' not in cls.__dict__
      # `__repr__` exists but is the default dataclass implementation
      or cls.__repr__.__qualname__ == '__create_fn__.<locals>.__repr__'
  )


def add_repr(cls: _ClsT) -> _ClsT:
  """Add a `.__repr__` method to the class, if not already present."""
  if has_default_repr(cls):
    cls.__repr__ = __repr__
  return cls


@reprlib.recursive_repr()
def __repr__(self) -> str:  # pylint: disable=invalid-name
  """Pretty-print `repr`."""
  all_fields = dataclasses.fields(self)
  collapse = len(all_fields) <= 1

  trailing = '' if collapse else ','

  lines = epy.Lines()
  lines += f'{self.__class__.__name__}('
  with lines.indent():
    for field in all_fields:
      if not field.repr:
        continue
      lines += f'{field.name}={getattr(self, field.name)!r}{trailing}'
  lines += ')'
  return lines.join(collapse=collapse)
