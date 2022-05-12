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

"""GPath wrapper around the gfile API."""

from __future__ import annotations

import ntpath
import os
import pathlib
import posixpath
import types
import typing
from typing import Any, ClassVar, Iterator, Optional, Type, TypeVar, Union

from etils import epy
from etils.epath import abstract_path
from etils.epath.typing import PathLike
import tensorflow as tf

_P = TypeVar('_P')

URI_PREFIXES = ('gs://', 's3://')
_URI_SCHEMES = frozenset(('gs', 's3'))

_URI_MAP_ROOT = {
    'gs://': '/gs/',
    's3://': '/s3/',
}

_PREFIX_TO_BACKEND = {}


class _GPath(abstract_path.Path):
  """Pathlib like api around `tf.io.gfile`."""

  # `_PATH` is `posixpath` or `ntpath`.
  # Use explicit `join()` rather than `super().joinpath()` to avoid infinite
  # recursion.
  # Do not use `os.path`, so `PosixGPath('gs://abc')` works on windows.
  _PATH: ClassVar[types.ModuleType]

  def __new__(cls: Type[_P], *parts: PathLike) -> _P:
    full_path = '/'.join(os.fspath(p) for p in parts)
    if full_path.startswith(URI_PREFIXES):
      prefix, _ = full_path.split('://', maxsplit=1)
      prefix = f'{prefix}://'
      new_prefix = _URI_MAP_ROOT[prefix]
      return super().__new__(cls, full_path.replace(prefix, new_prefix, 1))
    else:
      return super().__new__(cls, *parts)

  def _new(self: _P, *parts: PathLike) -> _P:
    """Create a new `Path` child of same type."""
    return type(self)(*parts)

  # Could try to use `cached_property` when beam is compatible (currently
  # raise mutable input error).
  @property
  def _uri_scheme(self) -> Optional[str]:
    if (len(self.parts) >= 2 and self.parts[0] == '/' and
        self.parts[1] in _URI_SCHEMES):
      return self.parts[1]
    else:
      return None

  @property
  def _path_str(self) -> str:
    """Returns the `__fspath__` string representation."""
    uri_scheme = self._uri_scheme
    if uri_scheme:  # pylint: disable=using-constant-test
      return self._PATH.join(f'{uri_scheme}://', *self.parts[2:])
    else:
      return self._PATH.join(*self.parts) if self.parts else '.'

  def __fspath__(self) -> str:
    return self._path_str

  def __str__(self) -> str:  # pylint: disable=invalid-str-returned
    return self._path_str

  def __repr__(self) -> str:
    return f'{type(self).__name__}({self._path_str!r})'

  def exists(self) -> bool:
    """Returns True if self exists."""
    return tf.io.gfile.exists(self._path_str)

  def is_dir(self) -> bool:
    """Returns True if self is a directory."""
    return tf.io.gfile.isdir(self._path_str)

  def iterdir(self: _P) -> Iterator[_P]:
    """Iterates over the directory."""
    for f in tf.io.gfile.listdir(self._path_str):
      yield self._new(self, f)

  def expanduser(self: _P) -> _P:
    """Returns a new path with expanded `~` and `~user` constructs."""
    return self._new(self._PATH.expanduser(self._path_str))

  def resolve(self: _P, strict: bool = False) -> _P:
    """Returns the abolute path."""
    return self._new(self._PATH.abspath(self._path_str))

  def glob(self: _P, pattern: str) -> Iterator[_P]:
    """Yielding all matching files (of any kind)."""
    for f in tf.io.gfile.glob(self._PATH.join(self._path_str, pattern)):
      yield self._new(f)

  def mkdir(
      self,
      mode: int = 0o777,
      parents: bool = False,
      exist_ok: bool = False,
  ) -> None:
    """Create a new directory at this given path."""
    if self.exists() and not exist_ok:
      raise FileExistsError(f'{self._path_str} already exists.')

    if parents:
      tf.io.gfile.makedirs(self._path_str)
    else:
      tf.io.gfile.mkdir(self._path_str)

  def rmdir(self) -> None:
    """Remove the empty directory."""
    if not self.is_dir():
      raise NotADirectoryError(f'{self._path_str} is not a directory.')
    if list(self.iterdir()):
      raise ValueError(f'Directory {self._path_str} is not empty')
    tf.io.gfile.rmtree(self._path_str)

  def rmtree(self) -> None:
    """Remove the directory."""
    tf.io.gfile.rmtree(self._path_str)

  def unlink(self, missing_ok: bool = False) -> None:
    """Remove this file or symbolic link."""
    try:
      tf.io.gfile.remove(self._path_str)
    except tf.errors.NotFoundError as e:
      if missing_ok:
        pass
      else:
        raise FileNotFoundError(str(e))

  def open(
      self,
      mode: str = 'r',
      encoding: Optional[str] = None,
      errors: Optional[str] = None,
      **kwargs: Any,
  ) -> typing.IO[Union[str, bytes]]:
    """Opens the file."""
    if errors:
      raise NotImplementedError
    if encoding and not encoding.lower().startswith(('utf8', 'utf-8')):
      raise ValueError(f'Only UTF-8 encoding supported. Not: {encoding}')
    gfile = tf.io.gfile.GFile(self._path_str, mode, **kwargs)
    gfile = typing.cast(typing.IO[Union[str, bytes]], gfile)
    return gfile

  def rename(self: _P, target: PathLike) -> _P:
    """Rename file or directory to the given target."""
    # Note: Issue if WindowsPath and target is gs://. Rather than using `_new`,
    # `GPath.__new__` should dynamically return either `PosixGPath` or
    # `WindowsPath`, similarly to `pathlib.Path`.
    target = self._new(target)
    tf.io.gfile.rename(self._path_str, os.fspath(target))
    return target

  def replace(self: _P, target: PathLike) -> _P:
    """Replace file or directory to the given target."""
    target = self._new(target)
    tf.io.gfile.rename(self._path_str, os.fspath(target), overwrite=True)
    return target

  def copy(self: _P, dst: PathLike, overwrite: bool = False) -> _P:
    """Remove the directory."""
    # Could add a recursive=True mode
    dst = self._new(dst)
    tf.io.gfile.copy(self._path_str, os.fspath(dst), overwrite=overwrite)
    return dst


class PosixGPath(_GPath):
  """Pathlib like api around `tf.io.gfile`."""
  _PATH = posixpath


class WindowsGPath(pathlib.PureWindowsPath, _GPath):
  """Pathlib like api around `tf.io.gfile`."""
  _PATH = ntpath
