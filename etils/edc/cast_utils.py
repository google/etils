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

"""Auto-apply normalization to a dataclass fields."""

from __future__ import annotations

import dataclasses
import functools
import typing
from typing import TypeVar

from etils import epy
from etils.edc import field_utils
import typing_extensions
from typing_extensions import Annotated


_T = TypeVar('_T')
_FnT = TypeVar('_FnT')
_ClsT = TypeVar('_ClsT')

_IS_NORMALIZED = object()


if typing.TYPE_CHECKING:

  # TODO(b/254514368): Remove hack
  class _AutoCastMeta(type):

    def __getitem__(cls, value):
      return value

  class AutoCast(metaclass=_AutoCastMeta):
    pass


else:
  AutoCast = Annotated[_T, _IS_NORMALIZED]  # pytype: disable=invalid-typevar


def apply_auto_cast_to_field(cls: _ClsT) -> _ClsT:
  """Apply the auto-casting magic."""
  cls._edc_auto_casted = False  # pylint: disable=protected-access
  cls.__new__ = _wrap_new(cls.__new__)

  return cls


def _wrap_new(old_new_fn: _FnT) -> _FnT:
  """`__new__` decorator for `apply_auto_cast_to_field`."""

  @functools.wraps(old_new_fn)
  def new_new_fn(cls, *args, **kwargs):
    if old_new_fn is object.__new__:
      self = old_new_fn(cls)
    else:
      self = old_new_fn(cls, *args, **kwargs)

    # Already called, skipping initialization
    if cls.__dict__.get('_edc_auto_casted'):
      return self

    # First time, apply to all parent classes .
    for curr_cls in cls.mro():  # Apply to all parent classes
      _apply_auto_cast(curr_cls)

    cls._edc_auto_casted = True  # pylint: disable=protected-access
    return self

  return new_new_fn


def _apply_auto_cast(cls):
  """Apply the auto-casting magic to a single class."""
  if cls.__dict__.get('_edc_auto_casted', True):
    # Either:
    # This class is not a `@edc.dataclass` (but parent might)
    # This class is already processed
    return

  hints = _get_type_hints(cls, include_extras=True)
  fields = {f.name: f for f in dataclasses.fields(cls)}

  for name, hint in hints.items():
    if name not in cls.__annotations__:
      continue  # Only add typing from the current class
    # TODO(epot): Should create a typing parsing util.
    if typing_extensions.get_origin(hint) is not Annotated:
      continue
    if not any(a is _IS_NORMALIZED for a in hint.__metadata__):
      continue

    # TODO(epot): Support `Optional`
    hint_cls = hint.__origin__  # Unwrap the original type

    field = fields[name]
    if field.default_factory is not dataclasses.MISSING:
      raise ValueError(
          f'dataclass field {name} of {cls} cannot be both `AutoCast` and'
          ' `default_factory=`'
      )

    # TODO(epot): Propagate other field_kwargs (through likely not necessary)
    cast_field = field_utils.field(validate=hint_cls)
    setattr(cls, name, cast_field)  # cls.__dict__[name] = cast_field
    cast_field.__set_name__(cls, name)  # Notify the descriptor


# Could merge this function with the one in `dataclass_array` in a util.
def _get_type_hints(cls, *, include_extras: bool = False):
  """`get_type_hints` with better error reporting."""
  # At this point, `ForwardRef` should have been resolved.
  try:
    return typing_extensions.get_type_hints(cls, include_extras=include_extras)
  except Exception as e:  # pylint: disable=broad-except
    msg = (
        f'Could not infer typing annotation of {cls.__qualname__} '
        f'defined in {cls.__module__}:\n'
    )
    lines = [f' * {k}: {v!r}' for k, v in cls.__annotations__.items()]
    lines = '\n'.join(lines)

    epy.reraise(e, prefix=msg + lines + '\n')
