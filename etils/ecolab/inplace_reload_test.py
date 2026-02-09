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

  def reimport_module(self, module_name: str, content: str) -> types.ModuleType:
    file = self.tmp_path / f'{module_name}.py'
    file.write_text(textwrap.dedent(content))

    with self.reloader.update_old_modules(
        reload=[module_name],
        verbose=True,
        reload_mode=ecolab.ReloadMode.UPDATE_INPLACE,
        recursive=True,
    ):
      return __import__(module_name)


def test_reload_instance(reloader: _Reloader):  # pylint: disable=redefined-outer-name
  old_module = reloader.reimport_module(
      'test_module',
      """
      class A:
        X = 'old'

        def fn(self):
          return 1
      """,
  )

  a = old_module.A()

  assert a.fn() == 1
  assert a.X == 'old'

  new_module = reloader.reimport_module(
      'test_module',
      """
      class A:
        X = 'new'

        def fn(self):
          return 2
      """,
  )

  assert old_module.A is new_module.A

  assert a.fn() == 2
  assert a.X == 'new'
  assert a.__class__ is new_module.A
  assert isinstance(a, old_module.A)
  assert isinstance(a, new_module.A)


def test_reload_alias(reloader: _Reloader):  # pylint: disable=redefined-outer-name
  old_module = reloader.reimport_module(
      'test_module',
      """
      class A:
        X = 'old'

        def fn(self):
          return 1

      Alias = A
      """,
  )

  a = old_module.Alias()

  new_module = reloader.reimport_module(
      'test_module',
      """
      class A:
        X = 'new'

        def fn(self):
          return 2
      """,
  )

  assert a.fn() == 2
  assert a.__class__ is new_module.A


def test_reload_dependant(reloader: _Reloader):  # pylint: disable=redefined-outer-name
  old_module = reloader.reimport_module(
      'test_module',
      """
      class B:
        def fn(self):
          return 1

      class A:
        X = B()

        def fn(self):
          return self.X.fn()
      """,
  )

  a = old_module.A()
  assert a.fn() == 1

  new_module = reloader.reimport_module(
      'test_module',
      """
      class B:
        def fn(self):
          return 2

      class A:
        X = B()

        def fn(self):
          return self.X.fn()
      """,
  )

  assert a.fn() == 2
  assert old_module.A is new_module.A


def test_reload_delete_class(reloader: _Reloader):  # pylint: disable=redefined-outer-name
  old_module = reloader.reimport_module(
      'test_module',
      """
      class B:
        def fn(self):
          return 1

      class A:
        X = B()

        def fn(self):
          return self.X.fn()
      """,
  )

  a = old_module.A()
  assert a.fn() == 1

  new_module = reloader.reimport_module(
      'test_module',
      """
      class A:
        def fn(self):
          return 2
      """,
  )
  assert not hasattr(new_module, 'B')
  assert old_module.A is new_module.A
  assert a.fn() == 2


def test_reload_delete_field(reloader: _Reloader):  # pylint: disable=redefined-outer-name
  old_module = reloader.reimport_module(
      'test_module',
      """
      class A:
        X = 1

        def fn(self):
          return self.X
      """,
  )

  a = old_module.A()
  assert a.fn() == 1

  reloader.reimport_module(
      'test_module',
      """
      class A:
        def fn(self):
          return 2
      """,
  )
  assert not hasattr(a, 'X')
  assert a.fn() == 2


def test_reload_decorated_method(reloader: _Reloader):  # pylint: disable=redefined-outer-name
  old_module = reloader.reimport_module(
      'test_module',
      """
      class A:
        def fn(self, value):
          return value
      """,
  )

  a = old_module.A()
  assert a.fn('value') == 'value'

  reloader.reimport_module(
      'test_module',
      """
      def decorator(fn):
        def wrapper(self, value, *args, **kwargs):
          return fn(self, 'decorated-' + value, *args, **kwargs)
        return wrapper

      class A:
        @decorator
        def fn(self, value):
          return value
      """,
  )
  assert a.fn('value') == 'decorated-value'


def test_reload_inter_module(reloader: _Reloader):  # pylint: disable=redefined-outer-name
  old_mod_a = reloader.reimport_module(
      'test_module_a',
      """
      class A:
        def fn(self):
          return 1
      """,
  )

  mod_b = reloader.reimport_module(
      'test_module_b',
      """
      import test_module_a

      inst = test_module_a.A()
      """,
  )

  assert mod_b.inst.fn() == 1

  new_mod_a = reloader.reimport_module(
      'test_module_a',
      """
      class A:
        def fn(self):
          return 2
      """,
  )

  assert mod_b.inst.fn() == 2
  assert mod_b.inst.__class__ is new_mod_a.A
  assert isinstance(mod_b.inst, old_mod_a.A)
  assert isinstance(mod_b.inst, new_mod_a.A)


# TODO(epot): Fix
@pytest.mark.skip('%autoreload implementation do not support cycles')
def test_cycles(reloader: _Reloader):  # pylint: disable=redefined-outer-name
  # Test reload when there's cycles
  old_module = reloader.reimport_module(
      'test_module',
      """
      class A:
        pass

      A.A = A
      """,
  )

  old_A = old_module.A  # pylint: disable=invalid-name

  new_module = reloader.reimport_module(
      'test_module',
      """
      class A:
        pass

      A.A = A
      """,
  )

  assert old_A is not new_module.A
  assert old_A.A is new_module.A
  assert old_module.A is new_module.A


def test_reload_enum(reloader: _Reloader):  # pylint: disable=redefined-outer-name
  old_mod = reloader.reimport_module(
      'test_module',
      """
      import enum

      from etils import epy


      class A(enum.Enum):
        FOO = enum.auto()
        BAR = enum.auto()

      class B(epy.StrEnum):
        FOO = enum.auto()
        BAR = enum.auto()
      """,
  )

  a = old_mod.A.FOO
  b = old_mod.B.BAR

  # Reload enums with a new field (not in order!).
  mod = reloader.reimport_module(
      'test_module',
      """
      import enum

      from etils import epy


      class A(enum.Enum):
        BUZZ = enum.auto()

        FOO = enum.auto()
        BAR = enum.auto()

      class B(epy.StrEnum):
        BUZZ = enum.auto()

        FOO = enum.auto()
        BAR = enum.auto()
      """,
  )

  # Still equal after reload.
  assert a == mod.A.FOO
  assert mod.A.FOO == a
  # And doesn't radomly compare to other enum values.
  assert a != mod.A.BAR
  # repr enum types still behave.
  assert b != mod.B.FOO
  assert b == mod.B.BAR
  assert 'foo' == mod.B.FOO
  # Enum fieds get added.
  assert hasattr(mod.A, 'BUZZ')
  # The singletons are updated (even if saved values aren't).
  assert old_mod.A.FOO is mod.A.FOO
