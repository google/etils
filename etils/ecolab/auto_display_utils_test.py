# Copyright 2023 The etils Authors.
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
import textwrap

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
    assert auto_display_utils._has_trailing_semicolon(code.splitlines(), stmt)[
        0
    ]

  code = epy.dedent("""
  a=1
  b = 3
  c=3  # asdasd
  (
      a  # aa
  )=(
      1,
  3,)  # asdasd
  a=1;b=2
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
    )[0]

  code = epy.dedent("""
  a=1;
  b=2;
  """)
  node = ast.parse(code)

  assert auto_display_utils._has_trailing_semicolon(
      code.splitlines(), node.body[0]
  ) == (True, False)
  assert auto_display_utils._has_trailing_semicolon(
      code.splitlines(), node.body[1]
  ) == (True, True)


@pytest.mark.parametrize(
    "code",
    [
        """
        b = 2
        other = 123;

        a = 1;
        """,
        """
        a=1;

        # Some comment
        # other comment
        """,
        """
        a=1;


        """,
        "a=1;  # Comment",
    ],
)
def test_ignore_last(code):
  code = textwrap.dedent(code)
  node = ast.parse(code)

  for stmt in node.body[:-1]:
    assert not auto_display_utils._has_trailing_semicolon(
        code.splitlines(), stmt
    )[1]
  assert auto_display_utils._has_trailing_semicolon(
      code.splitlines(), node.body[-1]
  )[1]


def test_ignore_last_not_last():
  code = textwrap.dedent("""
  if x:
    a = 1;
  """)
  node = ast.parse(code)

  has_trailing, is_last_statement = auto_display_utils._has_trailing_semicolon(
      code.splitlines(), node.body[-1].body[-1]  # pytype: disable=attribute-error
  )
  assert has_trailing
  assert not is_last_statement
