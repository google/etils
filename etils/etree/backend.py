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

"""Tree API backends."""

from __future__ import annotations

import abc
import types
from typing import Any, Callable, TypeVar

from etils import epy
from etils.etree.typing import Tree

_T = TypeVar('_T')
_Tin = Any  # TypeVar('_Tin')
_Tout = Any  # TypeVar('_Tout')

# Structure which allow to reconstruct the tree
# * jax: TreeDef
# * tf.nest/tree: The original Tree
_TreeDef = Any


class Backend(abc.ABC):
  """Tree API backend.

  Note: The backend lazy-import the module on first call. This
  allow to use `etree` with Jax even if TF isn't installed (and
  vice-versa).
  """

  @epy.cached_property
  def module(self) -> types.ModuleType:
    """Module used by the backend."""
    try:
      module = self.import_module()
    except ImportError as e:
      epy.reraise(e, suffix=f'etree backend require {self.MODULE_NAME!r}.')
    return module

  @abc.abstractmethod
  def import_module(self) -> types.ModuleType:
    """Import and return the module."""
    raise NotImplementedError

  @abc.abstractmethod
  def map(
      self,
      map_fn: Callable[..., _Tout],  # Callable[[_Tin0, _Tin1,...], Tout]
      *trees: Tree[_Tin],  # _Tin0, _Tin1,...
  ) -> Tree[_Tout]:
    """Like `tf.nest.map_structure`."""
    raise NotImplementedError

  @abc.abstractmethod
  def flatten(self, tree: Tree[_T]) -> tuple[list[_T], _TreeDef]:
    """Like `tf.nest.flatten`."""
    raise NotImplementedError

  @abc.abstractmethod
  def unflatten(self, structure: _TreeDef, flat_sequence: list[_T]) -> Tree[_T]:
    raise NotImplementedError

  @abc.abstractmethod
  def assert_same_structure(
      self,
      struct0: Tree[Any],
      struct1: Tree[Any],
  ) -> None:
    raise NotImplementedError


class Jax(Backend):
  """`jax.tree_util` backend."""

  def import_module(self):
    import jax  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error
    return jax.tree_util

  def map(self, map_fn, *trees):
    return self.module.tree_multimap(map_fn, *trees)

  def flatten(self, tree):
    flat_vals, treedef = self.module.tree_flatten(tree)
    return flat_vals, treedef

  def unflatten(self, structure, flat_sequence):
    return structure.unflatten(flat_sequence)

  def assert_same_structure(
      self,
      tree0: Tree[Any],
      tree1: Tree[Any],
  ):
    treedef0 = self.module.tree_structure(tree0)
    treedef1 = self.module.tree_structure(tree1)
    if treedef0 != treedef1:
      raise ValueError(
          "The two structures don't have the same nested structure.\n"
          f"Left: {treedef0}\nRight: {treedef1}")


class DmTree(Backend):
  """`tree` backend."""

  def import_module(self):
    import tree  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error
    return tree

  def map(self, map_fn, *trees):
    return self.module.map_structure(map_fn, *trees)

  def flatten(self, tree):
    return self.module.flatten(tree), tree

  def unflatten(self, structure, flat_sequence):
    return self.module.unflatten_as(structure, flat_sequence)

  def assert_same_structure(
      self,
      tree0: Tree[Any],
      tree1: Tree[Any],
  ):
    self.module.assert_same_structure(tree0, tree1)


class Nest(Backend):
  """`tf.nest` backend."""

  def import_module(self):
    import tensorflow as tf  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error
    return tf.nest

  def map(self, map_fn, *trees):
    return self.module.map_structure(map_fn, *trees)

  def flatten(self, tree):
    return self.module.flatten(tree), tree

  def unflatten(self, structure, flat_sequence):
    return self.module.pack_sequence_as(structure, flat_sequence)

  def assert_same_structure(
      self,
      tree0: Tree[Any],
      tree1: Tree[Any],
  ):
    self.module.assert_same_structure(tree0, tree1)
