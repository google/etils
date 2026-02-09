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

from etils import epy
from etils.epy import lazy_api_imports_utils_test

with epy.lazy_api_imports(globals()):
  from etils.epy.env_utils import is_notebook  # pylint: disable=g-import-not-at-top,unused-import,g-importing-member


def test_imports():
  assert 'is_notebook' not in globals()  # Assert was removed from the module.
  # But is available through `__getattr__`.
  assert not lazy_api_imports_utils_test.is_notebook()
