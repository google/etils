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

"""Tests for flags."""

from absl import flags
from etils import epath


def test_parse_with_value():
  flagvalues = flags.FlagValues()
  epath.DEFINE_path('mypath', None, 'a path', flagvalues)
  flagvalues(['binary', '--mypath=/hello/world'])
  assert epath.Path('/hello/world') == flagvalues.mypath
