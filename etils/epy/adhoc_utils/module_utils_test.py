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

"""Test."""

import sys
import types

from etils.epy.adhoc_utils import module_utils
import pytest


@pytest.mark.parametrize(
    'in_, out',
    [
        (
            'etils.epy.adhoc_utils',  # Already a module
            'etils.epy.adhoc_utils',
        ),
    ],
)
def test_path_to_module_name(in_: str, out: str):
  assert module_utils.path_to_module_name(in_) == out


def test_get_module_names_respects_name_boundaries(monkeypatch):
  names = [
      '_etils_test_pkg',
      '_etils_test_pkg.child',
      '_etils_test_pkg_extra',
      '_etils_test_pkg_extra.child',
  ]
  for name in names:
    monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

  assert module_utils.get_module_names('_etils_test_pkg') == names[:2]
  assert (
      module_utils.get_module_names('_etils_test_pkg', recursive=False)
      == names[:1]
  )
  assert (
      module_utils.get_module_names(
          ['_etils_test_pkg', '_etils_test_pkg_extra']
      )
      == names
  )
  assert module_utils.get_module_names([]) == []


def test_clear_cached_modules_preserves_sibling_prefix(monkeypatch):
  name = '_etils_reload_test'
  sibling = types.ModuleType(name + '_extra')
  monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
  monkeypatch.setitem(
      sys.modules, name + '.child', types.ModuleType(name + '.child')
  )
  monkeypatch.setitem(sys.modules, sibling.__name__, sibling)

  module_utils.clear_cached_modules(name, invalidate=False)

  assert name not in sys.modules
  assert name + '.child' not in sys.modules
  assert sys.modules[sibling.__name__] is sibling
