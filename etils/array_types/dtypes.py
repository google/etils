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

"""Dtype utils."""

from __future__ import annotations

import abc
import typing
from typing import Any

from etils import epy
import numpy as np

if typing.TYPE_CHECKING:
  from etils import enp


# TODO(epot): Replace dtype by generic `AnyFloat`, `AnyInt` dtype.
_STD_TYPE_TO_NP_DTYPE = {
    int: np.int32,
    float: np.float32,
    bool: np.bool_,
}
_NP_KIND_TO_STR = {
    'u': 'ui',
    'i': 'i',
    'f': 'f',
    'V': 'bf',  # Because np.dtype(jnp.bfloat16).kind == 'V'
    'c': 'complex',
    'b': 'bool_',
    'U': 'str',  # Unicode
    'O': 'str',  # Unicode
}
# Numpy kinds which should be displayed with bits number (`f32`,...)
_BITS_KINDS = {'u', 'i', 'f', 'c', 'V'}


class DType(abc.ABC):
  """DType wrapper.

  This allow to support more complex types, like dtype unions.

  This is EXPERIMENTAL, so the API might change.

  """

  @classmethod
  def from_value(cls, value: Any) -> 'DType':
    """Convert the value to dtype."""
    if value is None:
      return AnyDType()
    if isinstance(value, DType):
      return value
    elif isinstance(value, np.dtype) or epy.issubclass(value, np.generic):
      return NpDType(value)
    if value in _STD_TYPE_TO_NP_DTYPE:
      return NpDType(_STD_TYPE_TO_NP_DTYPE[value])
    else:
      raise TypeError(f'Unsuported dtype: {value!r}')

  @property
  @abc.abstractmethod
  def array_cls_name(self) -> str:
    """Name of the array class associated with the dtype (`f32`, `ui8`,...)."""
    raise NotImplementedError

  @abc.abstractmethod
  def asarray(self, array_like, *, xnp: enp.NpModule):
    raise NotImplementedError

  @abc.abstractmethod
  def __eq__(self, other) -> bool:
    raise NotImplementedError

  @abc.abstractmethod
  def __hash__(self) -> int:
    raise NotImplementedError

  def __repr__(self) -> str:
    return f'DType({self.array_cls_name!r})'


class NpDType(DType):
  """Raw numpy dtype."""

  def __init__(self, dtype):
    self.np_dtype = np.dtype(dtype)  # Normalize the dtype

  @property
  def array_cls_name(self) -> str:
    np_dtype = self.np_dtype
    kind_str = _NP_KIND_TO_STR[np_dtype.kind]
    if np_dtype.kind in _BITS_KINDS:
      # Display with the size (ui8, f32,...)
      return f'{kind_str}{np_dtype.itemsize * 8}'
    else:
      return kind_str  # Raw types (str, bool_,...)

  def asarray(self, array_like, *, xnp: enp.NpModule):
    # from etils import enp  # pylint: disable=g-import-not-at-top
    # TODO(epot): Add a strict=True mode to prevent `int` -> `float` conversion
    # Normalize list,... as array
    return xnp.asarray(array_like, dtype=self.np_dtype)

  def __eq__(self, other) -> bool:
    _assert_isdtype(other)
    return isinstance(other, NpDType) and self.np_dtype == other.np_dtype

  def __hash__(self) -> int:
    return hash(self.np_dtype)


class AnyDType(DType):
  """DType which can represent any dtype."""

  @property
  def array_cls_name(self) -> str:
    return 'Array'

  def asarray(self, array_like, *, xnp: enp.NpModule):
    return xnp.asarray(array_like)

  def __eq__(self, other) -> bool:
    _assert_isdtype(other)
    return isinstance(other, AnyDType)

  def __hash__(self) -> int:
    return hash(type(self).__qualname__)


class DTypeUnion(DType):
  """Generic dtype (float, int,...)."""

  def asarray(self, array_like, *, xnp: enp.NpModule):
    raise NotImplementedError


def _assert_isdtype(dtype: Any) -> None:
  if not isinstance(dtype, DType):
    raise TypeError('etils DTypes can only be compared with other etils')
