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

"""`os.path` API backend."""

from __future__ import annotations

import abc
import glob as glob_lib
import os
import shutil
import typing
from typing import Any, NoReturn, Optional, Union

from etils.epath.typing import PathLike

# Available modes (from tensorflow/python/lib/io/file_io.py;l=55)
# Also exclude `+` as broken in gfile
_OPEN_MODES = ('r', 'w', 'a')


class Backend(abc.ABC):
  """Abstract backend class."""

  def open(
      self,
      path: PathLike,
      mode: str = 'r',
      *,
      encoding: Optional[str] = None,
      errors: Optional[str] = None,
      **kwargs: Any,
  ) -> typing.IO[Union[str, bytes]]:
    """`open` with argument checking."""
    if errors:
      raise NotImplementedError('`errors=` not supported in `open()`.')
    if encoding and not encoding.lower().startswith(('utf8', 'utf-8')):
      raise ValueError(f'Only UTF-8 encoding supported. Not: {encoding}')
    # TODO(epot): Could support `x` mode

    mode_without_b = mode.replace('b', '')
    if mode_without_b not in _OPEN_MODES:
      raise ValueError(f'mode={mode_without_b!r} is not one of {_OPEN_MODES}')
    if kwargs:
      raise NotImplementedError(
          f'kwargs {list(kwargs)}` not supported in `open()`.')
    return self._open(path, mode)

  @abc.abstractmethod
  def _open(
      self,
      path: PathLike,
      mode: str,
  ) -> typing.IO[Union[str, bytes]]:
    """`open`. Encoding should be utf-8."""
    raise NotImplementedError

  @abc.abstractmethod
  def exists(self, path: PathLike) -> bool:
    raise NotImplementedError

  @abc.abstractmethod
  def isdir(self, path: PathLike) -> bool:
    raise NotImplementedError

  @abc.abstractmethod
  def listdir(self, path: PathLike) -> list[str]:
    raise NotImplementedError

  @abc.abstractmethod
  def glob(self, path: PathLike) -> list[str]:
    raise NotImplementedError

  @abc.abstractmethod
  def makedirs(self, path: PathLike) -> None:
    raise NotImplementedError

  @abc.abstractmethod
  def mkdir(self, path: PathLike) -> None:
    raise NotImplementedError

  @abc.abstractmethod
  def rmtree(self, path: PathLike) -> None:
    raise NotImplementedError

  @abc.abstractmethod
  def remove(self, path: PathLike) -> None:
    raise NotImplementedError

  @abc.abstractmethod
  def rename(self, path: PathLike, dst: PathLike) -> None:
    raise NotImplementedError

  @abc.abstractmethod
  def replace(self, path: PathLike, dst: PathLike) -> None:
    raise NotImplementedError

  @abc.abstractmethod
  def copy(self, path: PathLike, dst: PathLike, overwrite: bool) -> None:
    raise NotImplementedError


class _OsPathBackend(Backend):
  """`os.path` backend."""

  def _open(
      self,
      path: PathLike,
      mode: str,
  ) -> typing.IO[Union[str, bytes]]:
    if 'b' in mode:
      encoding = None
    else:
      encoding = 'utf-8'
    return open(path, mode, encoding=encoding)

  def exists(self, path: PathLike) -> bool:
    return os.path.exists(path)

  def isdir(self, path: PathLike) -> bool:
    return os.path.isdir(path)

  def listdir(self, path: PathLike) -> list[str]:
    # GFile filter backup files per default.
    return [p for p in os.listdir(path) if not p.endswith('~')]

  def glob(self, path: PathLike) -> list[str]:
    return glob_lib.glob(path)

  def makedirs(self, path: PathLike) -> None:
    os.makedirs(path)

  def mkdir(self, path: PathLike) -> None:
    os.mkdir(path)

  def rmtree(self, path: PathLike) -> None:
    try:
      shutil.rmtree(path)
    except NotADirectoryError:
      self.remove(path)

  def remove(self, path: PathLike) -> None:
    try:
      os.remove(path)
    except IsADirectoryError:
      os.rmdir(path)

  def rename(self, path: PathLike, dst: PathLike) -> None:
    if self.exists(dst):
      raise FileExistsError(
          f'Cannot rename {path}. Destination {dst} already exists.')
    os.rename(path, dst)

  def replace(self, path: PathLike, dst: PathLike) -> None:
    if self.isdir(dst):
      raise IsADirectoryError(f'Cannot overwrite: {dst} is a directory')
    os.replace(path, dst)

  def copy(self, path: PathLike, dst: PathLike, overwrite: bool) -> None:
    if not overwrite and self.exists(dst):
      raise FileExistsError(f'{dst} already exists. Cannot copy {path}.')
    shutil.copyfile(path, dst)


class _TfBackend(Backend):
  """TensorFlow backend."""

  @property
  def tf(self):
    try:
      import tensorflow  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error
    except ImportError:
      raise ImportError(  # pylint: disable=raise-missing-from
          'To use epath.Path with gs://, TensorFlow should be installed.')
    return tensorflow

  @property
  def gfile(self):
    return self.tf.io.gfile

  def _open(
      self,
      path: PathLike,
      mode: str,
  ) -> typing.IO[Union[str, bytes]]:
    return self.gfile.GFile(path, mode)

  def exists(self, path: PathLike) -> bool:
    return self.gfile.exists(path)

  def isdir(self, path: PathLike) -> bool:
    return self.gfile.isdir(path)

  def listdir(self, path: PathLike) -> list[str]:
    return self.gfile.listdir(path)

  def glob(self, path: PathLike) -> list[str]:
    return self.gfile.glob(path)

  def makedirs(self, path: PathLike) -> None:
    self.gfile.makedirs(path)

  def mkdir(self, path: PathLike) -> None:
    try:
      self.gfile.mkdir(path)
    except self.tf.errors.NotFoundError as e:
      raise FileNotFoundError(str(e)) from None

  def rmtree(self, path: PathLike) -> None:
    try:
      self.gfile.rmtree(path)
    except self.tf.errors.NotFoundError as e:
      raise FileNotFoundError(str(e)) from None

  def remove(self, path: PathLike) -> None:
    try:
      self.gfile.remove(path)
    except self.tf.errors.FailedPreconditionError as e:  # Dir not empty
      raise OSError(str(e)) from None
    except self.tf.errors.NotFoundError as e:
      raise FileNotFoundError(str(e)) from None

  def rename(self, path: PathLike, dst: PathLike) -> None:
    try:
      self.gfile.rename(path, dst)
    except self.tf.errors.OpError as e:
      self._reraise_error(e)

  def replace(self, path: PathLike, dst: PathLike) -> None:
    try:
      self.gfile.rename(path, dst, overwrite=True)
    except self.tf.errors.OpError as e:
      self._reraise_error(e)

  def copy(self, path: PathLike, dst: PathLike, overwrite: bool) -> None:
    if overwrite and self.isdir(dst):  # For consistency with rename, replace
      raise IsADirectoryError(
          f'Cannot copy {path}. Destination {dst} is a directory') from None
    try:
      self.gfile.copy(path, dst, overwrite=overwrite)
    except self.tf.errors.OpError as e:
      self._reraise_error(e)

  def _reraise_error(self, e) -> NoReturn:
    """Reraise the TF error."""
    e_msg = str(e)
    if isinstance(e, self.tf.errors.FailedPreconditionError):
      if 'not a directory' in e_msg.lower():
        raise NotADirectoryError(e_msg) from None
      if 'is a directory' in e_msg.lower():
        raise IsADirectoryError(e_msg) from None
      else:
        raise OSError(e_msg) from None
    if isinstance(e, self.tf.errors.AlreadyExistsError):
      e_msg = str(e)
      if 'is a directory' in e_msg.lower():
        raise IsADirectoryError(e_msg) from None
      else:
        raise FileExistsError(e_msg) from None
    if isinstance(e, self.tf.errors.NotFoundError):
      raise FileNotFoundError(e_msg) from None
    else:
      raise  # pylint: disable=misplaced-bare-raise

tf_backend = _TfBackend()
os_backend = _OsPathBackend()
