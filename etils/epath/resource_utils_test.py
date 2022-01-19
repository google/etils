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

"""Tests for resource_utils."""

import io
import os
import zipfile

from etils import epath


def test_resource():
  path = epath.resource_path('etils.epath')
  assert path.exists()
  assert path.is_dir()
  assert 'README.md' in set(p.name for p in path.iterdir())

  path = path / 'README.md'
  assert path.exists()
  assert path.read_text()


def _make_zip_file() -> zipfile.ZipFile:
  """Returns an in-memory zip file."""
  data = io.BytesIO()
  zf = zipfile.ZipFile(data, 'w')
  zf.writestr('a.txt', b'content of a')
  zf.writestr('b/c.txt', b'content of c')
  zf.writestr('b/d/e.txt', b'content of e')
  zf.writestr('b/f.txt', b'content of f')
  zf.writestr('g/h/i.txt', b'content of i')
  zf.filename = 'alpharep.zip'
  return zf


def test_resource_path():
  path = epath.resource_utils.ResourcePath(_make_zip_file())
  assert isinstance(path, os.PathLike)
  assert path.joinpath('b/c.txt').read_text() == 'content of c'
  sub_dirs = list(path.joinpath('b').iterdir())
  assert len(sub_dirs) == 3
  for p in sub_dirs:  # Childs should be `ResourcePath` instances
    assert isinstance(p, epath.resource_utils.ResourcePath)

  # Forwarded to `Path` keep the resource.
  path = epath.Path(path)
  assert isinstance(path, epath.resource_utils.ResourcePath)

  assert path.joinpath() == path
  assert path.joinpath('abc', 'def.txt').name == 'def.txt'
