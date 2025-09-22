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

from __future__ import annotations

import collections
import dataclasses
import textwrap

from etils import epy
import fiddle as fdl


def test_dedent():
  text1 = textwrap.dedent(
      """\
      A(
         x=1,
      )"""
  )
  text2 = epy.dedent(
      """
      A(
         x=1,
      )
      """
  )
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

  assert text == epy.dedent(
      """
      dict(
          a=1,
          b=2,
      )
      """
  )


def test_lines_block():
  assert epy.Lines.make_block('A', {}) == 'A()'
  assert epy.Lines.make_block('A', {}, braces='[') == 'A[]'
  assert epy.Lines.make_block('A', {'x': 1}) == 'A(x=1)'
  assert epy.Lines.make_block('A', {'x': 1}, braces=('<', '>')) == 'A<x=1>'
  assert epy.Lines.make_block('', {'x': 1}, braces='{', equal=': ') == '{x: 1}'
  assert epy.Lines.make_block(
      'A',
      {
          'x': 111,
          'y': 222,
          'z': 333,
      },
  ) == epy.dedent(
      """
      A(x=111, y=222, z=333)
      """
  )

  assert epy.Lines.make_block('', ['a', 'b'], braces='[') == "['a', 'b']"


def test_lines_std():
  @dataclasses.dataclass
  class B:
    x: int = 1
    y: int = 2

  @dataclasses.dataclass
  class A:
    t: tuple[str, ...] = ()
    l: list[int] = dataclasses.field(default_factory=list)
    d: dict[str, int] = dataclasses.field(default_factory=dict)
    dc: B = dataclasses.field(default_factory=B)  # pytype: disable=invalid-annotation,name-error
    s: str = 'aaa'

  a = A(
      t=('aaaaaaaaaaaaaaaaaaaa', 'bbbbbbbbbbbbbbbbbbbb'),
      l=[1, 2, 3, 4],
      d={'aaaaaaaaaaaaaaaaaaaa': 1, 'bbbbbbbbbbbbbbbbbbbb': 1},
  )

  repr_ = epy.pretty_repr(a)  # pytype: disable=wrong-arg-types
  assert repr_ == epy.dedent("""
  A(
      t=(
          'aaaaaaaaaaaaaaaaaaaa',
          'bbbbbbbbbbbbbbbbbbbb',
      ),
      l=[1, 2, 3, 4],
      d={
          'aaaaaaaaaaaaaaaaaaaa': 1,
          'bbbbbbbbbbbbbbbbbbbb': 1,
      },
      dc=B(x=1, y=2),
      s='aaa',
  )
  """)


def test_pretty_repr_tuple():
  assert epy.pretty_repr((1,)) == '(1,)'
  assert epy.pretty_repr((1, 2)) == '(1, 2)'
  assert epy.pretty_repr([1]) == '[1]'
  assert epy.pretty_repr([1, 2]) == '[1, 2]'


def _fn_with_var_args(*args):  # pylint: disable=unused-argument
  return


def _fn_with_var_kwargs(arg1, kwarg1=None, **kwargs):  # pylint: disable=unused-argument
  return


def test_pretty_repr_ordereddict():
  obj = collections.OrderedDict(
      x=2,
      z=3,
      y={'aaaaaaaaaaaaaaaaaaaa': 1, 'bbbbbbbbbbbbbbbbbbbb': 1},
  )
  expected_repr = epy.dedent("""
  OrderedDict({
      'x': 2,
      'z': 3,
      'y': {
          'aaaaaaaaaaaaaaaaaaaa': 1,
          'bbbbbbbbbbbbbbbbbbbb': 1,
      },
  })
  """)
  assert epy.pretty_repr(obj) == expected_repr


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

  assert text == epy.dedent(
      """
      [
        a,
        b,
        [
          c,
          d,
        ],
      ]
      """
  )


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

  assert repr(A(a=A(), b=A(a=A(a=None)))) == epy.dedent(
      """
      A(
          a=A(),
          b=A(a=A(a=None)),
      )
      """
  )

  assert repr(A(a=A(), b=A(a=A(a=None, b=A())))) == epy.dedent(
      """
      A(
          a=A(),
          b=A(a=A(
              a=None,
              b=A(),
          )),
      )
      """
  )


def test_diff():
  a = ('xxx', 'aaaaaaaaaaaaaaaaaaaa', 'bbbbbbbbbbbbbbbbbbbb')
  b = ('aaaaaaaaaaaaaaaaaaaa', 'bbbbbbbbbbbbbbbbbbbbnn')
  assert epy.diff_str(a, b) == """  (
-     'xxx',
      'aaaaaaaaaaaaaaaaaaaa',
-     'bbbbbbbbbbbbbbbbbbbb',
+     'bbbbbbbbbbbbbbbbbbbbnn',
?                          ++

  )"""


def test_pprint_namedtuple():
  A = collections.namedtuple('A', ['x', 'y'])
  a = A(
      x='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
      y='bbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
  )
  text = textwrap.dedent("""\
      A(
          x='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
          y='bbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
      )""")
  assert epy.pretty_repr(a) == text
