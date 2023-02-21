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

"""Colab utils."""

from __future__ import annotations

import contextlib
import html
import io
import json as json_std
import typing
from typing import Iterator
import uuid

import IPython.display

if typing.TYPE_CHECKING:
  JsonValue = str | float | int | bool | None
  Json = JsonValue | dict[JsonValue, 'Json'] | list['Json']


@contextlib.contextmanager
def _collapse_std(
    *,
    name: str,
    redirect_fn,
) -> Iterator[None]:
  """Base colapsible implementation."""
  name = html.escape(name)
  f = io.StringIO()
  with redirect_fn(f):
    yield
  content = f.getvalue()
  content = html.escape(content)
  content = f'<pre><code>{content}</code></pre>'
  content = IPython.display.HTML(
      f'<details><summary>{name}</summary>{content}</details>'
  )
  IPython.display.display(content)


@contextlib.contextmanager
def _redirect_stdall(new_target: io.StringIO) -> Iterator[None]:
  with contextlib.redirect_stderr(new_target):
    with contextlib.redirect_stdout(new_target):
      yield


@contextlib.contextmanager
def collapse(name: str = '') -> Iterator[None]:
  """Capture stderr/stdout and display it in a collapsible block.

  Args:
    name: Name of the collapsible section.

  Yields:
    None
  """
  with _collapse_std(name=name, redirect_fn=_redirect_stdall):
    yield


def json(value: Json) -> None:
  """Display the Json `dict` / `list` interactivelly (with collapsible elems).

  Args:
    value: Json `dict` or `list` to inspect.
  """
  # Unique id to make sure multiple Json display do not interact with each other
  id_ = uuid.uuid1().hex

  html_content = html.escape(json_std.dumps(value))
  # There are a lot of alternative. Likely the most popular one is
  # `react-json-view`. However, this one display preview for collapsible
  # elements, which is nice (and not present in other alternatives).
  # https://github.com/mac-s-g/react-json-view/issues/237

  # TODO(epot): Could also add
  # `<input placeholder="Search" onkeydown="search(this)"></input>`
  # but likely not super useful.
  html_content = f"""
  <script src="https://unpkg.com/@alenaksu/json-viewer@2.0.0/dist/json-viewer.bundle.js"></script>
  <script>
    const viewer{id_} = document.querySelector('#json{id_}');
  </script>
  <button onclick="viewer{id_}.expandAll();">Expand All</button>
  <button onclick="viewer{id_}.collapseAll();">Collapse All</button>
  <json-viewer id="json{id_}">{html_content}</json-viewer>
  """
  IPython.display.display(IPython.display.HTML(html_content))
