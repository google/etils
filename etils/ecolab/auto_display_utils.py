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
import traceback
import typing
from typing import TypeVar

from etils import epy
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

  Note that `;` added to the last statement of the cell still silence the
  output (`IPython` default behavior).

  ```python
  x = my_fn();  # Display `my_fn()`

  my_fn();  # Display `my_fn()`
  ```

  `;` behavior can be disabled with `ecolab.auto_display(False)`

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
      return

else:

  class _RecordLines:
    """Record lines."""

    def __init__(self):
      self.last_lines = []

    def __call__(self, lines: list[str]) -> list[str]:
      self.last_lines = [l.rstrip('\n') for l in lines]
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
      has_trailing, is_last_statement = _has_trailing_semicolon(
          self.lines_recorder.last_lines, node
      )
      if has_trailing:
        if is_last_statement and isinstance(node, ast.Expr):
          # Last expressions are already displayed by IPython, so instead
          # IPython silence the statement
          pass
        elif node.value is None:  # `AnnAssign().value` can be `None` (`a: int`)
          pass
        else:
          node.value = ast.Call(
              func=_parse_expr('ecolab.auto_display_utils._display_and_return'),
              args=[node.value],
              keywords=[],
          )
    except Exception as e:
      code = '\n'.join(self.lines_recorder.last_lines)
      print(f'Error for code:\n-----\n{code}\n-----')
      traceback.print_exception(e)
      raise
    return node

  # pylint: disable=invalid-name
  visit_Assign = _maybe_display
  visit_AnnAssign = _maybe_display
  visit_Expr = _maybe_display
  visit_Return = _maybe_display
  # pylint: enable=invalid-name


def _parse_expr(code: str) -> ast.AST:
  return ast.parse(code, mode='eval').body


def _has_trailing_semicolon(
    code_lines: list[str],
    node: ast.AST,
) -> tuple[bool, bool]:
  """Check if `node` has trailing `;`."""
  if isinstance(node, ast.AnnAssign) and node.value is None:
    return False, False  # `AnnAssign().value` can be `None` (`a: int`)
  # Extract the lines of the statement
  last_line = code_lines[node.end_lineno - 1]  # lineno starts at `1`
  # Check if the last character is a `;` token
  has_trailing = False
  for char in last_line[node.end_col_offset :]:
    if char == ';':
      has_trailing = True
    elif char == ' ':
      continue
    elif char == '#':  # Comment,...
      break
    else:  # e.g. `a=1;b=2`
      has_trailing = False
      break

  is_last_statement = True  # Assume statement is the last one
  for line in code_lines[node.end_lineno :]:  # Next statements are all empty
    line = line.strip()
    if line and not line.startswith('#'):
      is_last_statement = False
      break
  if last_line.startswith(' '):
    # statement is inside `if` / `with` / ...
    is_last_statement = False
  return has_trailing, is_last_statement


def _display_and_return(x: _T) -> _T:
  """Print `x` and return `x`."""
  IPython.display.display(x)
  return x
