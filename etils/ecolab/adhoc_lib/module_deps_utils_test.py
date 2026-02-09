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

from etils.ecolab.adhoc_lib import module_deps_utils


def test_get_module_deps():
  import pytest
  pytest.skip('internal only')

  all_module_deps = module_deps_utils.get_all_module_deps()

  module_deps = all_module_deps['etils.ecolab.adhoc_lib.module_deps_utils']

  assert module_deps.name == 'etils.ecolab.adhoc_lib.module_deps_utils'
  assert set(module_deps.importing) == {
      'dataclasses',
      'inspect',
      'sys',
      'etils.epy',
      'types',
  }
  assert set(module_deps.imported_in) == {
      'etils.ecolab.adhoc_lib',
      'etils.ecolab.adhoc_lib.module_deps_utils_test',
      'etils.ecolab.adhoc_lib.reload_workspace_lib',
  }
