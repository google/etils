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

"""Tests."""

from __future__ import annotations

import dataclasses
import os
import pathlib
import sys
import textwrap
import types
from typing import Iterator

from etils import ecolab
from etils.ecolab import inplace_reload
import pytest

# TODO(epot): Fix open source tests
pytest.skip(allow_module_level=True)


@pytest.fixture
def reloader(tmp_path: pathlib.Path) -> Iterator[_Reloader]:
  """Fixture to import dynamically generated files."""
  tmp_path = tmp_path / 'imports'
  tmp_path.mkdir(parents=True, exist_ok=True)
  sys.path.append(os.fspath(tmp_path))
  yield _Reloader(tmp_path)
  sys.path.remove(os.fspath(tmp_path))


@dataclasses.dataclass(frozen=True)
class _Reloader:
  tmp_path: pathlib.Path
  reloader: inplace_reload._InPlaceReloader = dataclasses.field(
      default_factory=inplace_reload._InPlaceReloader
  )

  def reimport_module(self, content: str) -> types.ModuleType:
    file = self.tmp_path / 'inplace_test_file.py'
    file.write_text(textwrap.dedent(content))

    with self.reloader.update_old_modules(
        reload=['inplace_test_file'],
        verbose=True,
        reload_mode=ecolab.ReloadMode.UPDATE_INPLACE,
    ):
      import inplace_test_file  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

    return inplace_test_file


def test_reload_instance(reloader: _Reloader):  # pylint: disable=redefined-outer-name
  old_module = reloader.reimport_module("""

  class A:
    X = 'old'

    def fn(self):
      return 1

  """)

  a = old_module.A()

  assert a.fn() == 1
  assert a.X == 'old'

  # After reload, the new method is used

  new_module = reloader.reimport_module("""

  class A:
    X = 'new'

    def fn(self):
      return 2

  """)

  assert old_module.A is new_module.A

  assert a.fn() == 2
  assert a.X == 'new'
  assert a.__class__ is new_module.A
  assert isinstance(a, old_module.A)
  assert isinstance(a, new_module.A)


# TODO(epot): Fix
@pytest.mark.skip('%autoreload implementation do not support cycles')
def test_cycles(reloader: _Reloader):  # pylint: disable=redefined-outer-name
  # Test reload when there's cycles
  old_module = reloader.reimport_module("""

  class A:
    pass


  A.A = A
  """)

  old_A = old_module.A  # pylint: disable=invalid-name

  # After reload, the new method is used

  new_module = reloader.reimport_module("""

  class A:
    pass


  A.A = A
  """)

  assert old_A is not new_module.A
  assert old_A.A is new_module.A
  assert old_module.A is new_module.A


# TODO(epot): Restore
# def test_reload_enum(reloader: _Reloader):  # pylint: disable=redefined-outer-name
#   del reloader
