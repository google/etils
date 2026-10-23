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


def test_match():
  assert epy.reverse_fstring(
      '/home/{user}/projects/{project}',
      '/home/conchylicultor/projects/my-menhir',
  ) == {
      'user': 'conchylicultor',
      'project': 'my-menhir',
  }
  assert (
      epy.reverse_fstring(
          '/home/{user}/projects/{project}', '/home/conchylicultor/projects/'
      )
      is None
  )  # Empty `{project}` not supported
  assert epy.reverse_fstring(
      '/home/{user}/projects/{project}',
      '/home/user-projects/'
  ) is None


def test_literal_punctuation():
  assert epy.reverse_fstring('file.{ext}', 'file.txt') == {'ext': 'txt'}
  assert epy.reverse_fstring('file.{ext}', 'fileXtxt') is None

  assert epy.reverse_fstring('report[final]-{id}', 'report[final]-7') == {
      'id': '7'
  }
  assert epy.reverse_fstring('report[final]-{id}', 'reportf-7') is None

  assert epy.reverse_fstring('price${val}', 'price$10') == {'val': '10'}
  assert epy.reverse_fstring('price${val}', 'price10') is None
