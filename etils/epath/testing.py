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

"""Test utils for epath."""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import typing
from typing import Any, Callable, Iterator, Optional, Union
from unittest import mock

from etils.epath import backend
from etils.epath import gpath
from etils.epath import stat_utils
from etils.epath.typing import PathLike

_MockFn = Callable[..., Any]


_backend_cls = backend._OsPathBackend  # pylint: disable=protected-access


@dataclasses.dataclass(eq=False)
class _MockBackend(_backend_cls):
  """Backend with functions overwritten."""

  mock_fns: dict[str, _MockFn]

  def _get_fn(self, fn_name):
    original_fn = getattr(super(), fn_name)
    if self.mock_fns[fn_name] is not None:
      fn_to_call = self.mock_fns[fn_name]
      fn_to_call = functools.partial(fn_to_call, original_fn)
    else:
      fn_to_call = original_fn
    return fn_to_call

  def open(
      self,
      path: PathLike,
      mode: str,
  ) -> typing.IO[Union[str, bytes]]:
    return self._get_fn('open')(path, mode)

  def exists(self, path: PathLike) -> bool:
    return self._get_fn('exists')(path)

  def isdir(self, path: PathLike) -> bool:
    return self._get_fn('isdir')(path)

  def listdir(self, path: PathLike) -> list[str]:
    return self._get_fn('listdir')(path)

  def glob(self, path: PathLike) -> list[str]:
    return self._get_fn('glob')(path)

  def makedirs(self, path: PathLike, *, exist_ok: bool = False) -> None:
    return self._get_fn('makedirs')(path, exist_ok=exist_ok)

  def mkdir(self, path: PathLike, *, exist_ok: bool = False) -> None:
    return self._get_fn('mkdir')(path, exist_ok=exist_ok)

  def rmtree(self, path: PathLike) -> None:
    return self._get_fn('rmtree')(path)

  def remove(self, path: PathLike) -> None:
    return self._get_fn('remove')(path)

  def rename(self, path: PathLike, dst: PathLike) -> None:
    return self._get_fn('rename')(path, dst)

  def replace(self, path: PathLike, dst: PathLike) -> None:
    return self._get_fn('replace')(path, dst)

  def copy(self, path: PathLike, dst: PathLike, *, overwrite: bool) -> None:
    return self._get_fn('copy')(path, dst, overwrite=overwrite)

  def stat(self, path: PathLike) -> stat_utils.StatResult:
    return self._get_fn('stat')(path)


@contextlib.contextmanager
def mock_epath(
    *,
    copy: Optional[_MockFn] = None,
    exists: Optional[_MockFn] = None,
    glob: Optional[_MockFn] = None,
    isdir: Optional[_MockFn] = None,
    listdir: Optional[_MockFn] = None,
    makedirs: Optional[_MockFn] = None,
    mkdir: Optional[_MockFn] = None,
    open: Optional[_MockFn] = None,  # pylint: disable=redefined-builtin
    remove: Optional[_MockFn] = None,
    rename: Optional[_MockFn] = None,
    replace: Optional[_MockFn] = None,
    rmtree: Optional[_MockFn] = None,
    stat: Optional[_MockFn] = None,
) -> Iterator[None]:
  """Mock epath.

  Mock the file system by replacing the given function by their mock.
  Only the function passed are mocked.
  The mock function should have signature: `(original_fn, path)` + eventual
    args/kwargs for specific functions.

  Args:
    copy: New function (after mocking)
    exists: New function (after mocking)
    glob: New function (after mocking)
    isdir: New function (after mocking)
    listdir: New function (after mocking)
    makedirs: New function (after mocking)
    mkdir: New function (after mocking)
    open: New function (after mocking)
    remove: New function (after mocking)
    rename: New function (after mocking)
    replace: New function (after mocking)
    rmtree: New function (after mocking)
    stat: New function (after mocking)

  Yields:
    None
  """
  mock_fns = dict(
      open=open,
      copy=copy,
      rename=rename,
      exists=exists,
      glob=glob,
      isdir=isdir,
      listdir=listdir,
      makedirs=makedirs,
      mkdir=mkdir,
      remove=remove,
      replace=replace,
      rmtree=rmtree,
      stat=stat,
      # 'walk',
  )
  mock_backend = _MockBackend(mock_fns=mock_fns)

  # Replace all backend by the mock backend
  new_prefix_to_backend = {k: mock_backend for k in gpath._PREFIX_TO_BACKEND}  # pylint: disable=protected-access
  with mock.patch.object(gpath, '_PREFIX_TO_BACKEND', new_prefix_to_backend):
    yield
