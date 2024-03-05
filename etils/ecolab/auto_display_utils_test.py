# Copyright 2024 The etils Authors.
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

import ast

from etils import epy
from etils.ecolab import auto_display_utils
import pytest


def test_parsing():
  code = epy.dedent("""
  a: int=1;
  b.abc=3;
  c=3;  # asdasd
  (
      a  # aa
  )=(
      1,
  3,);  # asdasd
  assert fn();
  d  =  3   ;  # AAA
  (
      1,
      2,
  )  ;  # Comment
  (
      1,
      2,
  )  ;
  (
      1,
      2,
  );
  """)
  node = ast.parse(code)

  for stmt in node.body:
    assert auto_display_utils._has_trailing_semicolon(
        code.splitlines(), stmt
    ).has_trailing

  code = epy.dedent("""
  a=1
  b = 3
  ann: int;
  c=3  # asdasd
  (
      a  # aa
  )=(
      1,
  3,)  # asdasd
  a=1;b=2
  a=1;iiiy
  a=1 # ;a
  d  =  3     # AAA
  (
      1,
      2,
  )    # Comment
  (
      1,
      2,
  )
  (
      1,
      2,
  )
  """)
  node = ast.parse(code)

  for stmt in node.body:
    assert not auto_display_utils._has_trailing_semicolon(
        code.splitlines(), stmt
    ).has_trailing, f'Error for: `{ast.unparse(stmt)}`'


@pytest.mark.parametrize(
    'code, options',
    [
        ('x;', ''),
        ('x;#i', ''),
        ('x;a', 'a'),
        ('x;sp', 'sp'),
        ('x;ps', 'ps'),
        ('x;i', 'i'),
        ('x   ;i', 'i'),
        ('x   ;  i', 'i'),
        ('x; i', 'i'),
        ('x;  i  ', 'i'),
        ('x;i#Comment', 'i'),
        ('x;i    #Comment', 'i'),
        ('x;  i  # Comment', 'i'),
    ],
)
def test_alias(code: str, options: str):
  node = ast.parse(code)
  info = auto_display_utils._has_trailing_semicolon(
      code.splitlines(), node.body[0]
  )
  assert info.has_trailing
  assert info.options == {auto_display_utils._Options(o) for o in options}


@pytest.mark.parametrize(
    'code',
    [
        # 'x;aa',
        # 'x;ap',
        # 'x;spp',
        'x;a=1',
    ],
)
def test_noalias(code):
  node = ast.parse(code)
  assert not auto_display_utils._has_trailing_semicolon(
      code.splitlines(), node.body[0]
  ).has_trailing


@pytest.mark.parametrize(
    'line, extracted',
    [
        ('a=1', 'a'),
        ('a: int=1', 'a'),
        ('a, b = 1', '(a, b)'),
        ('a + 1', 'a + 1'),
        ('fn()', 'fn()'),
        # TODO(epot): Also support `a=b=2`
    ],
)
def test_unparse_line(line: str, extracted: str):
  node = ast.parse(line).body[0]
  assert auto_display_utils._unparse_line(node).value == extracted
