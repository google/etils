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

"""Tests for dataclass_flags."""

import dataclasses

from etils import eapp


@dataclasses.dataclass
class Args:
  """Args."""

  user: str
  verbose: bool = False


def test_make_flag_parser():
  flags_parser = eapp.make_flags_parser(Args)

  args = flags_parser(['', '--user=some_user'])
  assert args.user == 'some_user'
  assert not args.verbose

  args = flags_parser(['', '--user=my_user', '--verbose'])
  assert args.user == 'my_user'
  assert args.verbose
