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

"""Tests for text_utils."""

import textwrap

from etils import epy


def test_dedent():

  text1 = textwrap.dedent("""\
      A(
         x=1,
      )""")
  text2 = epy.dedent("""
      A(
         x=1,
      )
      """)
  assert text1 == text2


def test_lines():
  d = {'a': 1, 'b': 2}

  lines = epy.Lines()
  lines += 'dict('
  with lines.indent():
    for k, v in d.items():
      lines += f'{k}={v},'
  lines += ')'
  text = lines.join()

  assert text == epy.dedent("""
      dict(
          a=1,
          b=2,
      )
      """)


def test_lines_nested_indent():

  lines = epy.Lines(indent=2)
  lines += '['
  with lines.indent():
    lines += 'a,\nb,'
    lines += '['
    with lines.indent():
      lines += 'c,\nd,'
    lines += '],'
  lines += ']'
  text = lines.join()

  assert text == epy.dedent("""
      [
        a,
        b,
        [
          c,
          d,
        ],
      ]
      """)


def test_lines_collapse():

  class A(dict):

    def __repr__(self) -> str:
      lines = epy.Lines()
      collapse = len(self) <= 1

      lines += type(self).__name__ + '('
      with lines.indent():
        for k, v in self.items():
          suffix = '' if collapse else ','
          lines += f'{k}={v}' + suffix
      lines += ')'
      return lines.join(collapse=collapse)

  assert repr(A(a=A(), b=A(a=A(a=None)))) == epy.dedent("""
      A(
          a=A(),
          b=A(a=A(a=None)),
      )
      """)

  assert repr(A(a=A(), b=A(a=A(a=None, b=A())))) == epy.dedent("""
      A(
          a=A(),
          b=A(a=A(
              a=None,
              b=A(),
          )),
      )
      """)
