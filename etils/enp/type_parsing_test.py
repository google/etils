# Copyright 2025 The etils Authors.
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

"""Tests for type_parsing."""

from __future__ import annotations

from typing import Optional, List, Union

from etils.array_types import f32  # pylint: disable=g-multiple-import
from etils.enp import type_parsing
import pytest


@pytest.mark.parametrize(
    'hint, expected',
    [
        (int, [int]),
        (f32[''], [f32['']]),
        (Union[f32, int], [f32, int]),
        (Union[f32[''], int, None], [f32[''], int, None]),
        (Optional[f32['']], [f32[''], None]),
        (Optional[Union[f32[''], int]], [f32[''], int, None]),
        (List[int], [List[int]]),
        (f32[3, 3], [f32[3, 3]]),
    ],
)
def test_get_leaf_types(hint, expected):
  assert type_parsing.get_leaf_types(hint) == expected
