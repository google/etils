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

"""Inspect API entry point."""

from __future__ import annotations

import functools

from etils import epath
from etils.ecolab import pyjs_com
from etils.ecolab.inspects import nodes
import IPython.display


@pyjs_com.register_js_fn
def get_html_content(id_: str) -> str:
  """Returns the inner content of the block id.

  Is called the first time a block is expanded.

  Args:
    id_: Id of the block to load

  Returns:
    The html to add.
  """
  node = nodes.Node.from_id(id_)
  return node.inner_html


def inspect(obj: object) -> None:
  """Inspect all attributes of a Python object interactivelly.

  Args:
    obj: Any object to inspect (module, class, dict,...).
  """

  root = nodes.Node.from_obj(obj)

  html_content = IPython.display.HTML(
      f"""
      {_css_import()}
      {pyjs_com.js_import()}
      {_js_import()}

      <ul class="tree-root">
        {root.header_html}
      </ul>
      <script>
        load_content("{root.id}");
      </script>
      """
  )
  IPython.display.display(html_content)


def _static_path() -> epath.Path:
  """Path to the static resources (`.js`, `.css`)."""
  return epath.resource_path('etils') / 'ecolab' / 'inspects' / 'static'


# TODO(epot): Use gstatic to serve those files.
@functools.lru_cache()
def _css_import() -> str:
  """CSS content."""
  content = _static_path().joinpath('theme.css').read_text()
  return f'<style>{content}</style>'


@functools.lru_cache()
def _js_import() -> str:
  """Js content."""
  content = _static_path().joinpath('main.js').read_text()
  return f'<script>{content}</script>'
