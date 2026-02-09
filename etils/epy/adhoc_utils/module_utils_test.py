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
