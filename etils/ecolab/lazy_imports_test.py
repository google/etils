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

"""Tests for lazy_imports."""

import sys


def test_lazy_imports():
  from etils.ecolab.lazy_imports import jax  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

  assert jax.tree_map(lambda x: x + 1, [0, 1]) == [1, 2]
  assert 'jax' in sys.modules
