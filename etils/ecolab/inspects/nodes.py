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

"""HTML builder for Python objects."""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import html
from typing import Any, ClassVar, Generic, TypeVar
import uuid

from etils import enp
from etils.ecolab.inspects import attrs
from etils.ecolab.inspects import html_helper as H

_T = TypeVar('_T')


# All nodes are loaded globally and never cleared. Indeed, it's not possible
# to know when the Javascript is deleted (cell is cleared).
# It could create RAM issues by keeping object alive for too long. Could
# use weakref if this become an issue in practice.
_ALL_NODES: dict[str, Node] = {}


@dataclasses.dataclass
class Node:
  """Node base class.

  Each node correspond to a `<li>` element in the nested tree. When the node
  is expanded, `inner_html` is add the child nodes.

  Attributes:
    id: HTML id used to identify the node.
  """

  id: str = dataclasses.field(init=False)

  def __post_init__(self):
    # TODO(epot): Could likely have shorter/smarter uuids
    self.id = str(uuid.uuid1())
    _ALL_NODES[self.id] = self

  @classmethod
  def from_id(cls, id_: str) -> Node:
    """Returns the cached node from the HTML id."""
    return _ALL_NODES[id_]

  @classmethod
  def from_obj(cls, obj: object, *, name: str = '') -> ObjectNode:
    """Factory of a node from any object."""
    for sub_cls in [
        ObjectNode,
    ]:
      if isinstance(obj, sub_cls.MATCH_TYPES):
        break
    else:
      raise TypeError(f'Unexpected object {obj!r}.')

    return sub_cls(obj=obj, name=name)

  @property
  def header_html(self) -> str:
    """`<li>` one-line content."""
    raise NotImplementedError

  @property
  def inner_html(self) -> str:
    """Inner content when the item is expanded."""
    raise NotImplementedError

  def _li(self, *, clickable: bool = True) -> Callable[..., str]:
    """`<li>` section, called inside `header_html`."""
    if clickable:
      class_ = 'register-onclick'
    else:
      class_ = 'caret-invisible'

    def apply(*content):
      return H.li(id=self.id)(H.span(class_=['caret', class_])(*content))

    return apply


@dataclasses.dataclass
class ObjectNode(Node, Generic[_T]):
  """Any Python objects."""

  obj: _T
  name: str

  MATCH_TYPES: ClassVar[type[Any] | tuple[type[Any], ...]] = object

  @property
  def header_html(self) -> str:
    if self.is_root:
      prefix = ''
    else:
      prefix = f'{self.name}: '
    return self._li(clickable=not self.is_leaf)(
        H.span(class_=['key-main'])(f'{prefix}{self.header_repr}')
    )

  @property
  def inner_html(self) -> str:
    all_childs = [c.header_html for c in self.all_childs]
    return H.ul(class_=['collapsible'])(*all_childs)

  @property
  def all_childs(self) -> list[Node]:
    """Extract all attributes."""
    all_childs = [
        Node.from_obj(v, name=k) for k, v in attrs.get_attrs(self.obj).items()
    ]
    return all_childs

  @property
  def header_repr(self) -> str:
    return _obj_html_repr(self.obj)

  @property
  def is_root(self) -> bool:
    """Returns `True` if the node is top-level."""
    return not bool(self.name)

  @property
  def is_leaf(self) -> bool:
    """Returns `True` if the node cannot be recursed into."""
    return False


def _obj_html_repr(obj: object) -> str:
  """Returns the object representation."""
  if isinstance(obj, type(None)):
    type_ = 'null'
  elif isinstance(obj, (int, float)):
    type_ = 'number'
  elif isinstance(obj, bool):
    type_ = 'boolean'
  elif isinstance(obj, (str, bytes)):
    type_ = 'string'
    obj = _truncate_long_str(repr(obj))
  elif isinstance(obj, enp.lazy.LazyArray):
    type_ = 'number'
    obj = enp.ArraySpec.from_array(obj)
  elif isinstance(obj, attrs.ExceptionWrapper):
    type_ = 'error'
    obj = obj.e
  else:
    type_ = 'preview'
    try:
      obj = repr(obj)
    except Exception as e:  # pylint: disable=broad-except
      return _obj_html_repr(attrs.ExceptionWrapper(e))
    obj = _truncate_long_str(obj)

  if not isinstance(obj, str):
    obj = repr(obj)
    obj = html.escape(obj)
  return H.span(class_=[type_])(obj)


def _truncate_long_str(value: str) -> str:
  """Truncate long strings."""
  value = html.escape(value)
  # TODO(epot): Make the `...` clickable to allow expand the `str` dynamically
  # TODO(epot): Truncate multi line repr to single line
  if len(value) > 80:
    return value[:80] + '...'
  else:
    return value
