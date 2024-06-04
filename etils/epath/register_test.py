# Copyright 2024 The etils Authors.
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

"""Tests for register."""

import os
import pathlib
import sys

from etils import epath
import pytest


def _sanitize(*parts: epath.PathLike) -> str:
  full_path = '/'.join(os.fspath(p) for p in parts)
  return full_path.replace('repo://', '/repo/', 1)


@epath.register_path_cls('repo://')
class _RepoPath(pathlib.PurePosixPath):

  if sys.version_info < (3, 12):

    def __new__(cls, *parts: epath.PathLike) -> '_RepoPath':
      full_path = _sanitize(*parts)
      return super().__new__(cls, full_path)

  else:

    def __init__(self, *parts: epath.PathLike):
      full_path = _sanitize(*parts)
      super().__init__(full_path)


@pytest.mark.parametrize(
    'path',
    [
        epath.Path('repo://foo/bar'),
        epath.Path('repo://foo') / 'bar',
        epath.Path('repo://') / 'foo' / 'bar',
        epath.Path(_RepoPath('repo://foo/bar')),
    ],
)
def test_register(path: epath.Path):
  assert os.fspath(path) == '/repo/foo/bar'
  assert isinstance(path, _RepoPath)
  assert isinstance(epath.Path(path), _RepoPath)
  assert os.fspath(path.parent) == '/repo/foo'
