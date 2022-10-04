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

"""Tests for backend."""

from __future__ import annotations

import dataclasses
import os
import pathlib
from typing import Dict, Union

from etils import epath
from etils import epy
import pytest

with_subtests = epy.testing.with_subtests

_DIR_NAMES = [
    '.hidden',
    '~cached~',
    'cached~',
    '~cached',
    'folder',
    'other.parts',
    '统一码',  # Test unicode filenames
]
_FILE_NAMES = [
    'file.txt',
    '.hidden.txt',
    '~cached.txt~',
    'cached.txt~',
    '~cached.txt',
    '统一码.txt',
]
_NAMES = _DIR_NAMES + _FILE_NAMES


_NOT_EXIST = object()  # Sentinel

# dict[<filename>, <filename-content or None for directories>]
_FileDict = Dict[str, Union[None, bytes, object]]

# Open source issues
# * `file~` not filtered
# * `.remove` do not work with dir (TF):
#   * In Unix: raise FailedPrecondition
#   * In Mac: raise PermissionDeniedError
# * `.replace`:
#   * TF do not raise IsDirectoryError for folder -> folder replacement
# ...


def _test_open(
    backend: epath.backend.Backend,
    tmp_path: pathlib.Path,
):
  p = tmp_path / '.byte_file.统一码.txt~'

  with backend.open(p, 'wb') as f:
    f.write(b'abc')

  with backend.open(p, 'ab') as f:
    f.write(b'def')

  with backend.open(p, 'rb') as f:
    assert f.read() == b'abcdef'

  p = tmp_path / 'text file.txt'

  with backend.open(p, 'w') as f:
    f.write('abc统一码')

  with backend.open(p, 'a') as f:
    f.write('def')

  with backend.open(p, 'r') as f:
    assert f.read() == 'abc统一码def'

  # TODO(epot): Add test with non-utf-8 character to make
  # sure errors are consistents.


def _test_exists(
    backend: epath.backend.Backend,
    tmp_path: pathlib.Path,
):
  _make_default_path(tmp_path)
  for name in _NAMES:
    p = tmp_path / name
    assert backend.exists(p)
    assert backend.exists(os.fspath(p))

  assert not backend.exists(tmp_path / 'non-existing')
  assert not backend.exists(tmp_path / 'nested/non-existing')


def _test_isdir(
    backend: epath.backend.Backend,
    tmp_path: pathlib.Path,
):
  _make_default_path(tmp_path)

  for name in _DIR_NAMES:
    p = tmp_path / name
    assert backend.isdir(p)
    assert backend.isdir(os.fspath(p))

  for name in _FILE_NAMES:
    p = tmp_path / name
    assert not backend.isdir(p)
    assert not backend.isdir(os.fspath(p))

  assert not backend.isdir(tmp_path / 'non-existing')
  assert not backend.isdir(tmp_path / 'nested/non-existing')


def _test_glob(
    backend: epath.backend.Backend,
    tmp_path: pathlib.Path,
):
  (tmp_path / 'abc/nested/001').mkdir(parents=True)
  (tmp_path / 'abc/nested/002').mkdir(parents=True)
  (tmp_path / 'abc/nested/003').mkdir(parents=True)
  (tmp_path / 'abc/other_nested/001').mkdir(parents=True)
  (tmp_path / 'abc/other_nested/002').mkdir(parents=True)

  pattern = os.fspath(tmp_path / 'abc/*/001')
  assert sorted(backend.glob(pattern)) == [
      os.fspath(tmp_path / 'abc/nested/001'),
      os.fspath(tmp_path / 'abc/other_nested/001'),
  ]


def _test_listdir(
    backend: epath.backend.Backend,
    tmp_path: pathlib.Path,
):
  _make_default_path(tmp_path)

  all_names = set(_NAMES)
  all_names.remove('~cached~')  # tf.io.gfile filter backup files.
  all_names.remove('cached~')
  all_names.remove('~cached.txt~')
  all_names.remove('cached.txt~')
  assert sorted(backend.listdir(tmp_path)) == sorted(all_names)


def _test_makedirs(
    backend: epath.backend.Backend,
    tmp_path: pathlib.Path,
):
  for name in _DIR_NAMES:
    p = tmp_path / name / 'nested/other'
    backend.makedirs(p, exist_ok=True)
    assert backend.isdir(p)

  # Should be no-op when the directory already exists
  for name in _DIR_NAMES:
    p = tmp_path / name / 'nested'
    assert backend.isdir(p)
    assert backend.isdir(p / 'other')
    backend.makedirs(p, exist_ok=True)
    assert backend.isdir(p)
    assert backend.isdir(p / 'other')

  # Raise error when the directory is a file
  for name in _FILE_NAMES:
    p = tmp_path / name
    p.touch()
    with pytest.raises(FileExistsError):
      backend.makedirs(p, exist_ok=True)
    assert not backend.isdir(p)


def _test_makedirs_exists_not_ok(
    backend: epath.backend.Backend,
    tmp_path: pathlib.Path,
):
  for name in _DIR_NAMES:
    p = tmp_path / name / 'nested/other'
    backend.makedirs(p, exist_ok=False)
    assert backend.isdir(p)

  # Should raise error when the directory already exists
  for name in _DIR_NAMES:
    p = tmp_path / name / 'nested'
    assert backend.isdir(p)
    assert backend.isdir(p / 'other')
    with pytest.raises(FileExistsError):
      backend.makedirs(p, exist_ok=False)
    assert backend.isdir(p)
    assert backend.isdir(p / 'other')

  # Raise error when the directory is a file
  for name in _FILE_NAMES:
    p = tmp_path / name
    p.touch()
    with pytest.raises(FileExistsError):
      backend.makedirs(p, exist_ok=False)
    assert not backend.isdir(p)


def _test_mkdir(
    backend: epath.backend.Backend,
    tmp_path: pathlib.Path,
):
  for name in _DIR_NAMES:
    p = tmp_path / name
    backend.mkdir(p, exist_ok=True)
    assert backend.isdir(p)

  # When the file already exists
  for name in _DIR_NAMES:
    p = tmp_path / name
    backend.mkdir(p, exist_ok=True)
    assert backend.isdir(p)

  # Raise error when the directory is a file
  for name in _FILE_NAMES:
    p = tmp_path / name
    p.touch()
    with pytest.raises(FileExistsError):
      backend.mkdir(p, exist_ok=True)
    assert not backend.isdir(p)

  with pytest.raises(FileNotFoundError, match='No such file or directory'):
    backend.mkdir(tmp_path / 'nested/non-existing', exist_ok=True)


def _test_mkdir_exists_not_ok(
    backend: epath.backend.Backend,
    tmp_path: pathlib.Path,
):
  for name in _DIR_NAMES:
    p = tmp_path / name
    backend.mkdir(p, exist_ok=False)
    assert backend.isdir(p)

  # When the file already exists
  for name in _DIR_NAMES:
    p = tmp_path / name
    with pytest.raises(FileExistsError):
      backend.mkdir(p, exist_ok=False)
    assert backend.isdir(p)

  # Raise error when the directory is a file
  for name in _FILE_NAMES:
    p = tmp_path / name
    p.touch()
    with pytest.raises(FileExistsError):
      backend.mkdir(p, exist_ok=False)
    assert not backend.isdir(p)

  with pytest.raises(FileNotFoundError, match='No such file or directory'):
    backend.mkdir(tmp_path / 'nested/non-existing', exist_ok=False)


def _test_rmtree(
    backend: epath.backend.Backend,
    tmp_path: pathlib.Path,
):
  _make_default_path(tmp_path)

  for name in _NAMES:
    p = tmp_path / name
    assert p.exists()
    backend.rmtree(p)
    assert not p.exists()

  with pytest.raises(FileNotFoundError):
    backend.rmtree(tmp_path / 'non-existing')
  with pytest.raises(FileNotFoundError):
    backend.rmtree(tmp_path / 'nested/non-existing')


def _test_remove(
    backend: epath.backend.Backend,
    tmp_path: pathlib.Path,
):
  _make_default_path(tmp_path)

  for name in _FILE_NAMES:
    p = tmp_path / name
    backend.remove(p)
    assert not p.exists()

  for name in _DIR_NAMES:
    p = tmp_path / name
    backend.remove(p)
    assert not p.exists()

  # Non-empty directory
  with pytest.raises(OSError, match='Directory not empty'):
    (tmp_path / 'non-empty/nested/001').mkdir(parents=True)
    backend.remove(tmp_path / 'non-empty')

  with pytest.raises(FileNotFoundError):
    backend.remove(tmp_path / 'non-existing')
  with pytest.raises(FileNotFoundError):
    backend.remove(tmp_path / 'nested/non-existing')


def _test_rename(
    backend: epath.backend.Backend,
    tmp_path: pathlib.Path,
):
  # for make_src in _make_file, _make_folder, _make_nonempty_folder:
  for test_item in _SRC_DST_ITEMS:
    src_path, dst_path, dst_root_path = test_item.initialize(tmp_path)

    with epy.testing.subtest(dst_root_path.name):
      if isinstance(test_item.expected_rename, Exception):
        exc = test_item.expected_rename
        expected_msg = exc.args[0] if exc.args else None
        with pytest.raises(type(exc), match=expected_msg):
          backend.rename(src_path, dst_path)
      else:
        assert isinstance(test_item.expected_rename, dict)
        backend.rename(src_path, dst_path)
        expected = dict(test_item.expected_rename)
        expected['.'] = None
        assert _extract_dir_content(dst_root_path) == expected
        assert not src_path.exists()  # src path should not exists anymore

  # Test non-existing src
  with pytest.raises(FileNotFoundError):
    backend.rename(tmp_path / 'src-non-existing', tmp_path / 'dst-non-existing')


def _test_replace(
    backend: epath.backend.Backend,
    tmp_path: pathlib.Path,
):
  # for make_src in _make_file, _make_folder, _make_nonempty_folder:
  for test_item in _SRC_DST_ITEMS:
    src_path, dst_path, dst_root_path = test_item.initialize(tmp_path)

    with epy.testing.subtest(dst_root_path.name):
      if isinstance(test_item.expected_replace, Exception):
        exc = test_item.expected_replace
        expected_msg = exc.args[0] if exc.args else None
        with pytest.raises(type(exc), match=expected_msg):
          backend.replace(src_path, dst_path)
      else:
        assert isinstance(test_item.expected_replace, dict)
        backend.replace(src_path, dst_path)
        expected = dict(test_item.expected_replace)
        expected['.'] = None
        assert _extract_dir_content(dst_root_path) == expected
        assert not src_path.exists()  # src path should not exists anymore

  # Test non-existing src
  with pytest.raises(FileNotFoundError):
    backend.replace(
        tmp_path / 'src-non-existing',
        tmp_path / 'dst-non-existing',
    )


def _test_copy(
    backend: epath.backend.Backend,
    tmp_path: pathlib.Path,
):
  # for make_src in _make_file, _make_folder, _make_nonempty_folder:
  for test_item in _SRC_DST_ITEMS:
    src_path, dst_path, dst_root_path = test_item.initialize(tmp_path)

    if test_item.expected_copy is None:
      expected = test_item.expected_rename
    else:
      expected = test_item.expected_copy

    with epy.testing.subtest(dst_root_path.name):
      if isinstance(expected, Exception):
        exc = expected
        expected_msg = exc.args[0] if exc.args else None
        with pytest.raises(type(exc), match=expected_msg):
          backend.copy(src_path, dst_path, overwrite=False)
      else:
        assert isinstance(expected, dict)
        backend.copy(src_path, dst_path, overwrite=False)
        expected = dict(expected)
        expected['.'] = None
        assert _extract_dir_content(dst_root_path) == expected
        assert src_path.exists()  # src path still exists after copy

  # Test non-existing src
  with pytest.raises(FileNotFoundError):
    backend.copy(
        tmp_path / 'src-non-existing',
        tmp_path / 'dst-non-existing',
        overwrite=False,
    )


def _test_copy_with_overwrite(
    backend: epath.backend.Backend,
    tmp_path: pathlib.Path,
):
  # for make_src in _make_file, _make_folder, _make_nonempty_folder:
  for test_item in _SRC_DST_ITEMS:
    src_path, dst_path, dst_root_path = test_item.initialize(tmp_path)

    if test_item.expected_copy_overwrite is None:
      expected = test_item.expected_replace  # Copy+overwrite is like replace
    else:
      expected = test_item.expected_copy_overwrite

    with epy.testing.subtest(dst_root_path.name):
      if isinstance(expected, Exception):
        exc = expected
        expected_msg = exc.args[0] if exc.args else None
        with pytest.raises(type(exc), match=expected_msg):
          backend.copy(src_path, dst_path, overwrite=True)
      else:
        assert isinstance(expected, dict)
        backend.copy(src_path, dst_path, overwrite=True)
        expected = dict(expected)
        expected['.'] = None
        assert _extract_dir_content(dst_root_path) == expected
        assert src_path.exists()  # src path should still exist

  # Test non-existing src
  with pytest.raises(FileNotFoundError):
    backend.copy(
        tmp_path / 'src-non-existing',
        tmp_path / 'dst-non-existing',
        overwrite=True,
    )


def _test_stat(
    backend: epath.backend.Backend,
    tmp_path: pathlib.Path,
):
  _make_default_path(tmp_path)

  for name in _DIR_NAMES:
    p = tmp_path / name
    st_gt = os.stat(p)
    st = backend.stat(p)
    assert st.is_directory
    assert st.length == st_gt.st_size
    assert st.mtime == int(st_gt.st_mtime)

  for name in _FILE_NAMES:
    p = tmp_path / name
    st_gt = os.stat(p)
    st = backend.stat(p)
    assert not st.is_directory
    assert st.length == st_gt.st_size
    assert st.mtime == int(st_gt.st_mtime)


@pytest.mark.usefixtures('with_subtests')
@pytest.mark.parametrize(
    'test_fn',
    [
        _test_open,
        _test_exists,
        _test_isdir,
        _test_listdir,
        _test_glob,
        _test_makedirs,
        _test_makedirs_exists_not_ok,
        _test_mkdir,
        _test_mkdir_exists_not_ok,
        _test_rmtree,
        _test_remove,
        _test_rename,
        _test_replace,
        _test_copy,
        _test_copy_with_overwrite,
        _test_stat,
    ],
)
def test_backend(
    test_fn,
    tmp_path: pathlib.Path,
):
  backend_out = {}
  backends = {
      'os': epath.backend.os_backend,
      # Due to remaining small inconsistencies, do not test open source
      # tf backend
  }
  for backend_name, backend in backends.items():
    # Test each backend in a self contained dir
    backend_path = tmp_path / backend_name
    backend_path.mkdir()
    with epy.testing.subtest(backend_name):
      test_fn(backend, backend_path)
    backend_out[backend_name] = _extract_dir_content(backend_path)

  # Validate that all backend behave similarly
  (first_k, first_v), *others = backend_out.items()
  for k, v in others:
    assert first_v == v, f'{first_k} != {k}'


def _extract_dir_content(backend_path: pathlib.Path) -> _FileDict:
  out = {}
  _extract_dir_content_inner(out, backend_path, backend_path)
  return out


def _extract_dir_content_inner(
    out: _FileDict,
    root: pathlib.Path,
    curr_path: pathlib.Path,
) -> None:
  rel_name = os.fspath(curr_path.relative_to(root))
  if curr_path.is_file():
    out[rel_name] = curr_path.read_bytes()
  elif curr_path.is_dir():
    out[rel_name] = None
    # Recurse:
    for f in curr_path.iterdir():
      _extract_dir_content_inner(out, root, f)
  else:
    raise ValueError


def _write_dir_content(root_path: pathlib.Path, content: _FileDict) -> None:
  for k, v in content.items():
    p = root_path / k
    if v is None:  # Directory
      p.mkdir(parents=True, exist_ok=True)
    elif v is _NOT_EXIST:
      pass
    else:
      p.parent.mkdir(parents=True, exist_ok=True)
      p.write_bytes(v)


def _make_default_path(tmp_path: pathlib.Path):
  for name in _DIR_NAMES:
    (tmp_path / name).mkdir()
  for name in _FILE_NAMES:
    (tmp_path / name).write_text('abc')


# Utils to test rename, replace, copy


@dataclasses.dataclass
class _TestItem:
  """Item to test various combinaison of files copy, rename, replace."""

  src_file: _FileDict
  dst_file: _FileDict
  expected_rename: Union[Exception, _FileDict]
  expected_replace: Union[Exception, _FileDict]
  expected_copy: Union[Exception, _FileDict, None] = None
  expected_copy_overwrite: Union[Exception, _FileDict, None] = None

  def initialize(
      self,
      root_path: pathlib.Path,
  ) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    src_name, *_ = self.src_file  # pylint: disable=unpacking-non-sequence
    dst_name, *_ = self.dst_file  # pylint: disable=unpacking-non-sequence

    def _escape(s):
      s = s.replace('/', '-')
      return s

    esc_src_name = _escape(src_name)
    esc_dst_name = _escape(dst_name)

    src_dir = f'src-{esc_src_name}-{esc_dst_name}'
    dst_dir = f'dst-{esc_src_name}-{esc_dst_name}'

    src_root_dir = root_path / src_dir
    dst_root_dir = root_path / dst_dir
    src_root_dir.mkdir()
    dst_root_dir.mkdir()

    _write_dir_content(src_root_dir, self.src_file)
    _write_dir_content(dst_root_dir, self.dst_file)

    src_path = src_root_dir / src_name
    dst_path = dst_root_dir / dst_name
    return src_path, dst_path, dst_root_dir


# TODO(epot): Test rename in parent dir ?
# TODO(epot): Test rename with it-self ?
_SRC_DST_ITEMS = (
    # Rename/replace/copy a file
    _TestItem(
        src_file={'file.txt': b'abc'},
        dst_file={'file.txt': b'edf'},
        expected_rename=FileExistsError(),
        expected_replace={'file.txt': b'abc'},
    ),
    _TestItem(
        src_file={'file.txt': b'abc'},
        dst_file={'folder': None},
        expected_rename=FileExistsError(),
        expected_replace=IsADirectoryError(),
    ),
    _TestItem(
        src_file={'file.txt': b'abc'},
        dst_file={
            'non-empty-folder': None,
            'non-empty-folder/some_file': b'def',
        },
        expected_rename=FileExistsError(),
        expected_replace=IsADirectoryError(),
    ),
    _TestItem(
        src_file={'file.txt': b'abc'},
        dst_file={'nonexistent': _NOT_EXIST},
        expected_rename={'nonexistent': b'abc'},
        expected_replace={'nonexistent': b'abc'},
    ),
    _TestItem(
        src_file={'file.txt': b'abc'},
        dst_file={'nonexistent/nested': _NOT_EXIST},
        expected_rename=FileNotFoundError(),  # dst parent don't exists
        expected_replace=FileNotFoundError(),
    ),
    # Rename/replace/copy an empty folder
    _TestItem(
        src_file={'folder0': None},
        dst_file={'file.txt': b'edf'},
        expected_rename=FileExistsError(),
        expected_replace=NotADirectoryError(),
        expected_copy_overwrite=IsADirectoryError(),
    ),
    _TestItem(
        src_file={'folder0': None},
        dst_file={'folder': None},
        expected_rename=FileExistsError(),
        # GFile can't overwride directory, even thought `os` can
        expected_replace=IsADirectoryError(),
        expected_copy_overwrite=IsADirectoryError(),
    ),
    _TestItem(
        src_file={'folder0': None},
        dst_file={
            'non-empty-folder': None,
            'non-empty-folder/some_file': b'def',
        },
        expected_rename=FileExistsError(),
        expected_replace=IsADirectoryError(),
        expected_copy_overwrite=IsADirectoryError(),
    ),
    _TestItem(
        src_file={'folder0': None},
        dst_file={'nonexistent': _NOT_EXIST},
        expected_rename={'nonexistent': None},
        expected_replace={'nonexistent': None},
        # Only files can be copied
        expected_copy=IsADirectoryError(),
        expected_copy_overwrite=IsADirectoryError(),
    ),
    _TestItem(
        src_file={'folder0': None},
        dst_file={'nonexistent/nested': _NOT_EXIST},
        expected_rename=FileNotFoundError(),  # dst parent don't exists
        expected_replace=FileNotFoundError(),
        # Only files can be copied
        expected_copy=IsADirectoryError(),
        expected_copy_overwrite=IsADirectoryError(),
    ),
    # Rename/replace/copy an non-empty folder
    _TestItem(
        src_file={
            'folderfull': None,
            'folderfull/f.txt': b'abc',
        },
        dst_file={'file.txt': b'edf'},
        expected_rename=FileExistsError(),
        expected_replace=NotADirectoryError(),
        expected_copy_overwrite=IsADirectoryError(),
    ),
    _TestItem(
        src_file={
            'folderfull': None,
            'folderfull/f.txt': b'abc',
        },
        dst_file={'folder': None},
        expected_rename=FileExistsError(),
        expected_replace=IsADirectoryError(),
    ),
    _TestItem(
        src_file={
            'folderfull': None,
            'folderfull/f.txt': b'abc',
        },
        dst_file={
            'non-empty-folder': None,
            'non-empty-folder/some_file': b'def',
        },
        expected_rename=FileExistsError(),
        expected_replace=IsADirectoryError(),
    ),
    _TestItem(
        src_file={
            'folderfull': None,
            'folderfull/f.txt': b'abc',
        },
        dst_file={'nonexistent': _NOT_EXIST},
        expected_rename={
            'nonexistent': None,
            'nonexistent/f.txt': b'abc',
        },
        expected_replace={
            'nonexistent': None,
            'nonexistent/f.txt': b'abc',
        },
        # Only files can be copied
        expected_copy=IsADirectoryError(),
        expected_copy_overwrite=IsADirectoryError(),
    ),
    _TestItem(
        src_file={
            'folderfull': None,
            'folderfull/f.txt': b'abc',
        },
        dst_file={'nonexistent/nested': _NOT_EXIST},
        expected_rename=FileNotFoundError(),  # dst parent don't exists
        expected_replace=FileNotFoundError(),  # dst parent don't exists
        # Only files can be copied
        expected_copy=IsADirectoryError(),
        expected_copy_overwrite=IsADirectoryError(),
    ),
)
