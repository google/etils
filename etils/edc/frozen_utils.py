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

"""Unfrozen dataclasses."""

from __future__ import annotations

import dataclasses
from typing import Any, Generic, NoReturn, TypeVar, Union

from etils import epy

_Cls = TypeVar('_Cls')
_T = TypeVar('_T')


def add_unfrozen(cls: _Cls) -> _Cls:
  """Add the `frozen`, `unfrozen` methods."""
  cls_frozen = getattr(cls, 'frozen', None)
  cls_unfrozen = getattr(cls, 'has_unfrozen', None)

  # Already inherit a unfrozeen class
  if cls_frozen is frozen and cls_unfrozen is unfrozen:
    return cls

  # Partial implementation, or collision detected
  if cls_frozen is not None or cls_unfrozen is not None:
    raise ValueError(f'{cls} already define `frozen` or `unfrozen`')

  cls.frozen = frozen
  cls.unfrozen = unfrozen

  return cls


def frozen(self: _T) -> _T:
  """Freeze the dataclass."""
  raise ValueError('`.frozen()` can only be called after `.unfrozen()`.')


def unfrozen(self: _T) -> _T:
  """Returns a lazy deep-copy of the dataclass."""
  impl = _MutableProxyImpl(obj=self, common=_Common(), is_root=True)
  return impl.public_api


# Limitations: Compatibility of _MutableProxy with jax.tree_utils, chex,...


class _MutableProxy(Generic[_T]):
  """Proxy which mutate the dataclass.

  Note: To avoid attribute collisions with the wrapped class, the actual
  implementation is moved inside the `__impl` attribute.

  This module only expose the public API.

  """

  def __init__(self, impl: _MutableProxyImpl):
    super().__setattr__('_MutableProxy__impl', impl)

  def unfrozen(self) -> NoReturn:
    raise ValueError('Object is already unfrozen. Cannot call `.unfrozen()`.')

  def frozen(self) -> _T:
    return self.__impl.frozen()

  def __getattr__(self, name: str) -> Any:
    return self.__impl.getattr(name)

  def __setattr__(self, name: str, value: Any) -> None:
    return self.__impl.setattr(name, value)

  def __repr__(self) -> str:
    return f'{self.__class__.__name__}({self.__impl.obj})'


@dataclasses.dataclass
class _Common:
  """Shared variable across all nested childs of an `unfrozen()` object.

  Attributes:
    cache: Global mapping `id(_MutableProxyImpl) -> _MutableProxyImpl` to avoid
      duplicating the same object proxy  ```python a = a.unfrozen() a.x = x a.y
      = x  # `a.x` and `a.y` point to the same object a = a.frozen() assert a.x
      is a.y ```
    is_frozen: Become `True` after `.frozen()` is called. After which all
      Mutable are invalid
  """
  cache: dict[int, _MutableProxyImpl] = dataclasses.field(default_factory=dict)
  is_frozen: bool = False

  def get_proxy(self, value: Any) -> _MutableProxyImpl:
    """Returns the proxy associated with the given value, or create it."""
    id_ = id(value)
    if id_ not in self.cache:
      self.cache[id_] = _MutableProxyImpl(obj=value, common=self)
    return self.cache[id_]


@dataclasses.dataclass
class _MutableProxyImpl(Generic[_T]):
  """Proxy implementation is a separate class to avoid collisions."""
  obj: _T
  common: _Common
  is_root: bool = False

  # Child info
  attrs: dict[str, Union[_MutableProxyImpl,
                         Any]] = dataclasses.field(default_factory=dict)

  @epy.cached_property
  def public_api(self) -> _MutableProxy:
    return _MutableProxy(self)

  @epy.cached_property
  def _fields(self) -> dict[str, dataclasses.Field]:
    # Could also filter only init=True fields
    return {f.name: f for f in dataclasses.fields(self.obj)}

  def _is_dataclass_field(self, name: str) -> bool:
    """Returns True if the field is a dataclass attribute."""
    return name in self._fields  # pylint: disable=unsupported-membership-test

  def getattr(self, name: str) -> Any:
    """Returns `obj.name`."""
    if self.common.is_frozen:
      raise AttributeError('Cannot access value after the mutable was frozen.')

    # Reuse cache if it exists
    if name in self.attrs:
      value = self.attrs[name]
    else:
      value = getattr(self.obj, name)

      # Eventually wrap the value in a proxy
      if self._is_dataclass_field(name) and dataclasses.is_dataclass(value):
        value = self.common.get_proxy(value)
        self.attrs[name] = value

    if isinstance(value, _MutableProxyImpl):
      value = value.public_api
    return value

  def setattr(self, name: str, value: Any) -> None:
    """Set `obj.name`."""
    if self.common.is_frozen:
      raise AttributeError(
          'Cannot set attributes after the mutable was frozen.')

    # TODO(epot): Check that the field we're trying to overwrite is a
    # dataclass field
    if not self._is_dataclass_field(name):
      raise AttributeError(f'Cannot set {name!r}: Not a dataclass attribute.')

    # Trying to set another mutable mapping
    if isinstance(value, _MutableProxy):
      value = value._MutableProxy__impl  # pylint: disable=protected-access
      if value.common is not self.common:
        raise ValueError(
            f'Trying to mix `unfrozen` attributes. For: {name}={value}')

    # Wrapping dataclasses in proxy objects
    elif dataclasses.is_dataclass(value):
      value = self.common.get_proxy(value)

    # Storing the new value
    self.attrs[name] = value

  def frozen(self) -> _T:
    if not self.is_root:
      raise ValueError('Only the top-level dataclass can be `.frozen`')
    self.common.is_frozen = True

    return self.resolved

  # pytype: disable=invalid-annotation
  @epy.cached_property
  def resolved(self) -> _T:
  # pytype: enable=invalid-annotation
    """Recursivelly call `.replace` on instances which were mutated."""
    # Cached property, so that the same object is only resolved once
    new_vals = {}
    for k, v in self.attrs.items():
      if isinstance(v, _MutableProxyImpl):
        if v.obj is v.resolved:  # Skip attributes which were not mutated
          continue
        v = v.resolved
      new_vals[k] = v
    if not new_vals:  # Object wasn't mutated
      return self.obj
    return dataclasses.replace(self.obj, **new_vals)
