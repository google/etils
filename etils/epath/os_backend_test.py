# Copyright 2026 The etils Authors.
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

"""Tests for local replacement semantics, including Windows directory entries."""

import os
import stat
import types
from unittest import mock

from etils.epath import backend
import pytest


@pytest.mark.parametrize('nonempty', [False, True])
def test_replace_directory_preserves_destination(tmp_path, nonempty):
  source = tmp_path / 'source'
  source.mkdir()
  if nonempty:
    (source / 'payload').write_bytes(b'source contents')
  destination = tmp_path / 'destination'
  destination.write_bytes(b'destination contents')

  with pytest.raises(NotADirectoryError):
    backend.os_backend.replace(source, destination)

  assert destination.read_bytes() == b'destination contents'
  assert source.is_dir()
  if nonempty:
    assert (source / 'payload').read_bytes() == b'source contents'


@pytest.mark.parametrize('nonempty', [False, True])
def test_replace_directory_to_unused_path(tmp_path, nonempty):
  source = tmp_path / 'source'
  source.mkdir()
  if nonempty:
    (source / 'payload').write_bytes(b'source contents')
  destination = tmp_path / 'destination'

  backend.os_backend.replace(source, destination)

  assert not source.exists()
  assert destination.is_dir()
  if nonempty:
    assert (destination / 'payload').read_bytes() == b'source contents'


@pytest.mark.parametrize('destination_exists', [False, True])
def test_replace_file(tmp_path, destination_exists):
  source = tmp_path / 'source'
  source.write_bytes(b'source contents')
  destination = tmp_path / 'destination'
  if destination_exists:
    destination.write_bytes(b'destination contents')

  backend.os_backend.replace(source, destination)

  assert not source.exists()
  assert destination.read_bytes() == b'source contents'


@pytest.mark.parametrize('broken', [False, True])
def test_replace_directory_symlink(tmp_path, broken):
  target = tmp_path / 'target'
  if not broken:
    target.mkdir()
    (target / 'payload').write_bytes(b'target contents')
  source = tmp_path / 'source'
  try:
    source.symlink_to(target, target_is_directory=True)
  except OSError as error:
    if getattr(error, 'winerror', None) == 1314:
      pytest.skip('Creating Windows symlinks requires a privilege')
    raise
  destination = tmp_path / 'destination'
  destination.write_bytes(b'destination contents')

  if os.name == 'nt':
    with pytest.raises(NotADirectoryError):
      backend.os_backend.replace(source, destination)
    assert destination.read_bytes() == b'destination contents'
    assert source.is_symlink()
  else:
    backend.os_backend.replace(source, destination)
    assert destination.is_symlink()
    assert os.readlink(destination) == str(target)
    assert not source.is_symlink()
  if not broken:
    assert (target / 'payload').read_bytes() == b'target contents'


@pytest.mark.skipif(os.name != 'nt', reason='Windows directory attributes')
@pytest.mark.parametrize('mode', [stat.S_IFDIR, stat.S_IFLNK])
def test_replace_windows_directory_attribute(tmp_path, mode):
  # A directory symlink (including a dangling one) has DIRECTORY on the entry
  # itself. Simulate its lstat result without requiring symlink privileges.
  source = tmp_path / 'source'
  source.write_bytes(b'source contents')
  destination = tmp_path / 'destination'
  destination.write_bytes(b'destination contents')
  source_stat = types.SimpleNamespace(
      st_mode=mode,
      st_file_attributes=stat.FILE_ATTRIBUTE_DIRECTORY
      | stat.FILE_ATTRIBUTE_REPARSE_POINT,
  )
  with mock.patch.object(os, 'lstat', return_value=source_stat):
    with pytest.raises(NotADirectoryError):
      backend.os_backend.replace(source, destination)

  assert destination.read_bytes() == b'destination contents'
  assert source.read_bytes() == b'source contents'
