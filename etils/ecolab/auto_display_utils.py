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

"""Auto display statements ending with `;` on Colab."""

from __future__ import annotations

import ast
import dataclasses
import functools
import re
import traceback
import typing
from typing import TypeVar

from etils import epy
from etils.ecolab import array_as_img
from etils.ecolab.inspects import core as inspects
from etils.etree import jax as etree  # pylint: disable=g-importing-member
import IPython
import packaging

_T = TypeVar('_T')

# Old API (before IPython 7.0)
_IS_LEGACY_API = packaging.version.parse(
    IPython.__version__
) < packaging.version.parse('7')


def auto_display(activate: bool = True) -> None:
  """Activate auto display statements ending with `;` (activated by default).

  Add a trailing `;` to any statement (assignment, expression, return
  statement) to display the current line.
  This call `IPython.display.display()` for pretty display.

  This change the default IPython behavior where `;` added to the last statement
  of the cell still silence the output.

  ```python
  x = my_fn();  # Display `my_fn()`

  my_fn();  # Display `my_fn()`
  ```

  `;` behavior can be disabled with `ecolab.auto_display(False)`

  Format:

  *   `my_obj;`: Alias for `IPython.display.display(x)`
  *   `my_obj;s`: (`spec`) Alias for
      `IPython.display.display(etree.spec_like(x))`
  *   `my_obj;i`: (`inspect`) Alias for `ecolab.inspect(x)`
  *   `my_obj;a`: (`array`) Alias for `media.show_images(x)` /
      `media.show_videos(x)` (`ecolab.auto_plot_array` behavior)
  *   `my_obj;q`: (`quiet`) Don't display the line (e.g. last line)

  Args:
    activate: Allow to disable `auto_display`
  """
  if not epy.is_notebook():  # No-op outside Colab
    return

  # Register the transformation
  ip = IPython.get_ipython()
  shell = ip.kernel.shell

  # Clear previous transformation to support reload and disable.
  _clear_transform(shell.ast_transformers)
  if _IS_LEGACY_API:
    _clear_transform(shell.input_transformer_manager.python_line_transforms)
    _clear_transform(shell.input_splitter.python_line_transforms)
  else:
    _clear_transform(ip.input_transformers_post)

  if not activate:
    return

  # Register the transformation.
  # The `ast` do not contain the trailing `;` information, and there's no way
  # to access the original source code from `shell.ast_transformers`, so
  # the logic has 2 steps:
  # 1) Record the source code string using `python_line_transforms`
  # 2) Parse and add the `display` the source code with `ast_transformers`

  lines_recorder = _RecordLines()

  if _IS_LEGACY_API:
    _append_transform(
        shell.input_splitter.python_line_transforms, lines_recorder
    )
    _append_transform(
        shell.input_transformer_manager.python_line_transforms, lines_recorder
    )
  else:
    _append_transform(ip.input_transformers_post, lines_recorder)
  _append_transform(
      shell.ast_transformers, _AddDisplayStatement(lines_recorder)
  )


def _clear_transform(list_):
  # Support `ecolab` adhoc `reload`
  for elem in list(list_):
    if hasattr(elem, '__is_ecolab__'):
      list_.remove(elem)


def _append_transform(list_, transform):
  """Append `transform` to `list_` (with `ecolab` reload support)."""
  transform.__is_ecolab__ = True  # Marker to support `ecolab` adhoc `reload`
  list_.append(transform)


if _IS_LEGACY_API and not typing.TYPE_CHECKING:

  class _RecordLines(IPython.core.inputtransformer.InputTransformer):
    """Record lines."""

    def __init__(self):
      self._lines = []
      self.last_lines = []

      self.trailing_stmt_line_nums = {}
      super().__init__()

    def push(self, line):
      """Add a line."""
      self._lines.append(line)
      return line

    def reset(self):
      """Reset."""
      if self._lines:  # Save the last recorded lines
        # Required because reset is called multiple times on empty output,
        # so always keep the last non-empty output
        self.last_lines = []
        for line in self._lines:
          self.last_lines.extend(line.split('\n'))
      self._lines.clear()

      self.trailing_stmt_line_nums.clear()
      return

else:

  class _RecordLines:
    """Record lines."""

    def __init__(self):
      self.last_lines = []
      # Additional state (reset at each cell) to keep track of which lines
      # contain trailing statements
      self.trailing_stmt_line_nums = {}

    def __call__(self, lines: list[str]) -> list[str]:
      self.last_lines = [l.rstrip('\n') for l in lines]
      self.trailing_stmt_line_nums = {}
      return lines


class _AddDisplayStatement(ast.NodeTransformer):
  """Transform the `ast` to add the `IPython.display.display` statements."""

  def __init__(self, lines_recorder: _RecordLines):
    self.lines_recorder = lines_recorder
    super().__init__()

  def _maybe_display(
      self, node: ast.Assign | ast.AnnAssign | ast.Expr
  ) -> ast.AST:
    """Wrap the node in a `display()` call."""
    try:
      if self._is_alias_stmt(node):  # Alias statements should be no-op
        return ast.Pass()

      line_info = _has_trailing_semicolon(self.lines_recorder.last_lines, node)
      if line_info.has_trailing:
        if node.value is None:  # `AnnAssign().value` can be `None` (`a: int`)
          pass
        else:

          fn_name = _ALIAS_TO_DISPLAY_FN[line_info.alias].__name__

          node.value = ast.Call(
              func=_parse_expr(f'ecolab.auto_display_utils.{fn_name}'),
              args=[node.value],
              keywords=[],
          )
          self.lines_recorder.trailing_stmt_line_nums[line_info.line_num] = (
              line_info
          )
    except Exception as e:
      code = '\n'.join(self.lines_recorder.last_lines)
      print(f'Error for code:\n-----\n{code}\n-----')
      traceback.print_exception(e)
      raise
    return node

  def _is_alias_stmt(self, node: ast.AST) -> bool:
    match node:
      case ast.Expr(value=ast.Name(id=name)):
        pass
      case _:
        return False
    if name not in _ALIAS_TO_DISPLAY_FN:
      return False
    # The alias is not in the same line as a trailing `;`
    if node.end_lineno - 1 not in self.lines_recorder.trailing_stmt_line_nums:
      return False
    return True

  # pylint: disable=invalid-name
  visit_Assign = _maybe_display
  visit_AnnAssign = _maybe_display
  visit_Expr = _maybe_display
  visit_Return = _maybe_display
  # pylint: enable=invalid-name


def _parse_expr(code: str) -> ast.AST:
  return ast.parse(code, mode='eval').body


@dataclasses.dataclass(frozen=True)
class _LineInfo:
  has_trailing: bool
  alias: str
  line_num: int


def _has_trailing_semicolon(
    code_lines: list[str],
    node: ast.AST,
) -> _LineInfo:
  """Check if `node` has trailing `;`."""
  if isinstance(node, ast.AnnAssign) and node.value is None:
    return _LineInfo(
        has_trailing=False,
        alias='',
        line_num=-1,
    )  # `AnnAssign().value` can be `None` (`a: int`)

  # Extract the lines of the statement
  line_num = node.end_lineno - 1
  last_line = code_lines[line_num]  # lineno starts at `1`

  # `node.end_col_offset` is in bytes, so UTF-8 characters count 3.
  last_part_of_line = last_line.encode('utf-8')
  last_part_of_line = last_part_of_line[node.end_col_offset :]
  last_part_of_line = last_part_of_line.decode('utf-8')

  # Check if the last character is a `;` token
  has_trailing = False
  alias = ''
  if match := _detect_trailing_regex().match(last_part_of_line):
    has_trailing = True
    if match.group(1):
      alias = match.group(1)

  return _LineInfo(
      has_trailing=has_trailing,
      alias=alias,
      line_num=line_num,
  )


@functools.cache
def _detect_trailing_regex() -> re.Pattern[str]:
  """Check if the last character is a `;` token."""
  # Match:
  # * `; a`
  # * `; a # Some comment`
  # * `; # Some comment`
  # Do not match:
  # * `; a; b`
  # * `; a=1`
  available_chars = ''.join(_ALIAS_TO_DISPLAY_FN)
  return re.compile(f' *; *([{available_chars}])? *(?:#.*)?$')


def _display_and_return(x: _T) -> _T:
  """Print `x` and return `x`."""
  IPython.display.display(x)
  return x


def _display_specs_and_return(x: _T) -> _T:
  """Print `x` and return `x`."""
  IPython.display.display(etree.spec_like(x))
  return x


def _inspect_and_return(x: _T) -> _T:
  """Print `x` and return `x`."""
  inspects.inspect(x)
  return x


def _display_array_and_return(x: _T) -> _T:
  """Print `x` and return `x`."""
  html = array_as_img.array_repr_html(x)
  if html is None:
    IPython.display.display(x)
  else:
    IPython.display.display(IPython.display.HTML(html))
  return x


def _return_quietly(x: _T) -> _T:
  """Return `x` without display."""
  return x


_ALIAS_TO_DISPLAY_FN = {
    '': _display_and_return,
    's': _display_specs_and_return,
    'i': _inspect_and_return,
    'a': _display_array_and_return,
    'q': _return_quietly,
}
