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

"""Communication between Python and Javascript.

Could be released as a separate module once it also support non-colab notebooks.
"""

import functools
import traceback
from typing import TypeVar

from etils import epath
import IPython.display

_FnT = TypeVar('_FnT')


def register_js_fn(fn: _FnT) -> _FnT:
  r"""Decorator to make a function callable from Javascript.

  Usage:

  In Python:

  ```python
  @register_js_fn
  def my_fn(*args, **kwargs):
    return {'x': 123}
  ```

  The function can then be called from Javascript:

  ```python
  IPython.display.HTML(f\"\"\"
    {pyjs_com.js_import()}
    <script>
      out = await call_python('my_fn', [arg0, arg1], {kwarg0: val0});
      out["x"] === 123;
    </script>
  \"\"\")
  ```

  Note that Javascript require the `pyjs_com.js_import()` statement to be
  present in the HTML from the cell.

  Args:
    fn: The Python function, can return any json-like value or dict

  Returns:
    The Python function, unmodified
  """

  def decorated(*args, **kwargs):
    try:
      out = fn(*args, **kwargs)
      # Wrap non-dict values inside JSON
      if not isinstance(out, dict):
        out = {'__etils_pyjs__': out}
      return IPython.display.JSON(out)
    except Exception as e:
      traceback.print_exception(e)
      raise

  # TODO(epot): Support Jypyter notebooks
  try:
    from google.colab import output  # pylint: disable=g-import-not-at-top
  except ImportError:
    pass
  else:
    output.register_callback(fn.__name__, decorated)

  return fn


# TODO(epot): Host on gstatic and dynamically generate the URL from the
# local file hash.
# Auto-detect adhoc import or add a flag to load locally modified `.js`
@functools.lru_cache()
def js_import() -> str:
  """`<script></script>` to import to add in the HTML."""
  path = epath.resource_path('etils') / 'ecolab/pyjs_com/py_js_com.js'
  return f'<script>{path.read_text()}</script>'
