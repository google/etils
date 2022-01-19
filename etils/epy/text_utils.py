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

"""Text utils."""

from __future__ import annotations

import contextlib
import dataclasses
import textwrap
from typing import Iterable, Iterator


@dataclasses.dataclass
class _Line:
  """Line item."""
  content: str
  indent_lvl: int
  indent_size: int


class Lines:
  """Util to build multi-line text.

  Usefull for pretty-print tools and human readable `__repr__`.

  Example:

  ```python
  d = {'a': 1, 'b': 2}

  lines = epy.Lines()
  lines += 'dict('
  with lines.indent():
    for k, v in d.items():
      lines += f'{k}={v},'
  lines += ')'
  text = lines.join()
  ```

  Output:

  ```
  dict(
      a=1,
      b=2,
  )
  ```

  """

  def __init__(self, *, indent: int = 4):
    self._lines: list[_Line] = []
    self._indent_size = indent
    self._indent_lvl = 0

  def append(self, line: str) -> None:
    """Append a new line `str`."""
    if not isinstance(line, str):
      raise TypeError(f'Lines should be added with `str`. Got {line!r}')

    self._lines.append(
        _Line(
            content=line,
            indent_lvl=self._indent_lvl,
            indent_size=self._indent_size,
        ),)

  def extend(self, iterable: Iterable[str]) -> None:
    """Append all the new line `str` from the iterable."""
    for line in iterable:
      self.append(line)

  def __iadd__(self, line: str) -> Lines:
    """Append a new line `str`."""
    self.append(line)
    return self

  @contextlib.contextmanager
  def indent(self) -> Iterator[None]:
    self._indent_lvl += 1
    try:
      yield
    finally:
      self._indent_lvl -= 1

  def join(self, *, collapse: bool = False) -> str:
    """Returns the lines.

    Args:
      collapse: If `True`, all lines are merged together in a single line.

    Returns:
      text: All lines merged together
    """
    lines = []
    for line in self._lines:
      content = line.content
      if not collapse:
        # Add the indentation to all the sub-lines
        indentation = ' ' * line.indent_lvl * line.indent_size
        content = textwrap.indent(content, indentation)
      lines.append(content)

    if collapse:
      token = ''
    else:
      token = '\n'
    return token.join(lines)


def dedent(text: str) -> str:
  r"""Wrapper around `textwrap.dedent` which also `strip()` the content.

  Before:

  ```python
  text = textwrap.dedent(
      \"\"\"\\
      A(
         x=1,
      )\"\"\"
  )
  ```

  After:

  ```python
  text = epy.dedent(
      \"\"\"
      A(
         x=1,
      )
      \"\"\"
  )
  ```

  Args:
    text: The text to dedent

  Returns:
    The dedented text
  """
  return textwrap.dedent(text).strip()
