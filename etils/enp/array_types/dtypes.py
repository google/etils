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
import dataclasses
import enum
from typing import Any, ClassVar, Optional, Union

from etils import epy
from etils.enp import numpy_utils
import numpy as np

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


class Casting(epy.StrEnum):
  """Casting mode when converting to array.

  Attributes:
    ALL: Allow all casting (e.g. `float`-> `int`)
    NONE: No casting allowed for existing `ndarray`. Array-like built-ins (list,
      int,...) are casted to the given dtype.
  """

  ALL = enum.auto()
  NONE = enum.auto()


def _make_array_cls_name(np_dtype: np.dtype) -> str:
  """Makes the array class name for the dtype."""
  kind_str = _NP_KIND_TO_STR[np_dtype.kind]
  if np_dtype.kind in _BITS_KINDS:
    # Display with the size (ui8, f32,...)
    return f'{kind_str}{np_dtype.itemsize * 8}'
  else:
    return kind_str  # Raw types (str, bool_,...)


@dataclasses.dataclass
class DType(abc.ABC):
  """DType wrapper.

  This allow to support more complex types, like dtype unions.

  This is EXPERIMENTAL, so the API might change.

  Attributes:
    name: Representation name (e.g. np.uint8, AnyFloat...)
    array_cls_name: Name of the array class associated with the dtype (`f32`,
      `ui8`,...).
  """

  name: str
  array_cls_name: str

  @classmethod
  def from_value(cls, value: Any) -> 'DType':
    """Convert the value to dtype."""
    if value is None:
      return AnyDType()
    elif isinstance(value, DType):
      return value
    elif numpy_utils.lazy.is_dtype(value):
      return NpDType(numpy_utils.lazy.as_dtype(value))
    elif value in _STD_TYPE_TO_DTYPE:
      return _STD_TYPE_TO_DTYPE[value]
    else:
      raise TypeError(f'Unsuported dtype: {value!r}')

  def asarray(
      self,
      array_like,
      *,
      xnp: numpy_utils.NpModule,
      casting: Union[Casting, str] = Casting.ALL,
  ):
    """Creates an `xnp.ndarray` from the `array_like`.

    Args:
      array_like: Any array-like
      xnp: Target numpy module
      casting: If `NONE`, prevent casting.

    Returns:
      array: The xnp array.
    """
    casting = Casting(casting)
    from_dtype = numpy_utils.lazy.dtype_from_array(array_like, strict=False)
    to_dtype = self._get_target_dtype(from_dtype)

    if casting == casting.NONE:  # When casting is invalid
      if to_dtype is None:
        pass  # Keep original casting
      elif from_dtype is None:  # Array-like (int, float, list,...)
        # TODO(epot): Could try to infer array types to prevent float -> int
        pass
      elif from_dtype != to_dtype:
        raise ValueError(
            f'Cannot cast {from_dtype} to {to_dtype} (casting={casting}).'
        )
    elif casting == casting.ALL:
      pass  # Always try to cast to dtype
    else:
      raise NotImplementedError(f'Unsupported casting {casting}')
    return xnp.asarray(array_like, dtype=to_dtype)

  @abc.abstractmethod
  def _get_target_dtype(
      self,
      from_dtype: Optional[np.dtype],
  ) -> Optional[np.dtype]:
    """Validate and normalize the numpy dtype.

    Args:
      from_dtype: DType of the array to cast

    Returns:
      to_dtype: DType of the array after casting
    """

  @abc.abstractmethod
  def __eq__(self, other) -> bool:
    raise NotImplementedError

  @abc.abstractmethod
  def __hash__(self) -> int:
    raise NotImplementedError

  def __repr__(self) -> str:
    return f'DType({self.name})'


class NpDType(DType):
  """Raw numpy dtype."""

  def __init__(self, dtype):
    self.np_dtype = np.dtype(dtype)  # Normalize the dtype
    super().__init__(
        name=str(self.np_dtype),
        array_cls_name=_make_array_cls_name(self.np_dtype),
    )

  def _get_target_dtype(
      self,
      from_dtype: Optional[np.dtype],
  ) -> Optional[np.dtype]:
    # Here, we could validate invalid casting, like:
    # * float -> int
    # * int -> bool
    return self.np_dtype

  def __eq__(self, other) -> bool:
    _assert_isdtype(other)
    return isinstance(other, NpDType) and self.np_dtype == other.np_dtype

  def __hash__(self) -> int:
    return hash(self.np_dtype)


class _SingletonDType(DType):
  """DType without arguments."""

  ARRAY_CLS_NAME: ClassVar[str]

  def __init__(self):
    super().__init__(
        name=self.__class__.__name__,
        array_cls_name=self.ARRAY_CLS_NAME,
    )

  def __eq__(self, other) -> bool:
    _assert_isdtype(other)
    return isinstance(other, type(self))

  def __hash__(self) -> int:
    return hash(type(self).__qualname__)


class AnyDType(_SingletonDType):
  """DType which can represent any dtype."""

  ARRAY_CLS_NAME = 'Array'

  def _get_target_dtype(
      self,
      from_dtype: Optional[np.dtype],
  ) -> Optional[np.dtype]:
    return None  # Keep original dtype


class AnyFloat(_SingletonDType):
  """Generic float dtype (float32, float64, bfloat16,...)."""

  ARRAY_CLS_NAME = 'FloatArray'

  def _get_target_dtype(
      self,
      from_dtype: Optional[np.dtype],
  ) -> Optional[np.dtype]:
    if from_dtype is None:
      return np.float32
    elif _is_float(from_dtype):
      return from_dtype
    else:  # int, bool,...
      # Could validate dtype to restrict too implicit casting
      return np.float32


class AnyInt(_SingletonDType):
  """Generic int dtype (int32, int64, uint8,...)."""

  ARRAY_CLS_NAME = 'IntArray'

  def _get_target_dtype(
      self,
      from_dtype: Optional[np.dtype],
  ) -> Optional[np.dtype]:
    if from_dtype is None:
      return np.int32
    elif _is_integer(from_dtype):
      return from_dtype
    else:  # float, bool,...
      # Could validate dtype to restrict too implicit casting
      return np.int32


_STD_TYPE_TO_DTYPE = {
    int: AnyInt(),
    float: AnyFloat(),
    bool: NpDType(np.bool_),
}


def _is_float(dtype: np.dtype) -> bool:
  """Validate the dtype is float."""
  # `V` to support bfloat16
  return np.issubdtype(dtype, np.floating) or dtype.kind == 'V'


def _is_integer(dtype: np.dtype) -> bool:
  """Validate the dtype is integer."""
  return np.issubdtype(dtype, np.integer)


def _assert_isdtype(dtype: Any) -> None:
  if not isinstance(dtype, DType):
    raise TypeError('etils DTypes can only be compared with other etils')
