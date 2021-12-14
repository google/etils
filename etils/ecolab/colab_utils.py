# Copyright 2021 The etils Authors.
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

import contextlib
import html
import io
from typing import Iterator

import IPython.display


@contextlib.contextmanager
def _collapse_std(
    *,
    name: str,
    redirect_fn,
) -> Iterator[None]:
  """Base colapsible implementation."""
  f = io.StringIO()
  with redirect_fn(f):
    yield
  content = f.getvalue()
  content = html.escape(content)
  content = f'<pre><code>{content}</code></pre>'
  content = IPython.display.HTML(
      f'<details><summary>{name}</summary>{content}</details>')
  IPython.display.display(content)


@contextlib.contextmanager
def collapse_stdout(name: str = 'stdout') -> Iterator[None]:
  """Capture stdout and display it in a collapsible block.

  Args:
    name: Name of the collapsible section.

  Yields:
    None
  """
  with _collapse_std(name=name, redirect_fn=contextlib.redirect_stdout):
    yield


@contextlib.contextmanager
def collapse_stderr(name: str = 'stderr') -> Iterator[None]:
  """Capture stderr and display it in a collapsible block.

  Args:
    name: Name of the collapsible section.

  Yields:
    None
  """
  with _collapse_std(name=name, redirect_fn=contextlib.redirect_stderr):
    yield


@contextlib.contextmanager
def _redirect_stdall(new_target: io.StringIO) -> Iterator[None]:
  with contextlib.redirect_stderr(new_target):
    with contextlib.redirect_stdout(new_target):
      yield


@contextlib.contextmanager
def collapse_stdall(name: str = 'stdout/stderr') -> Iterator[None]:
  """Capture stderr/stdout and display it in a collapsible block.

  Args:
    name: Name of the collapsible section.

  Yields:
    None
  """
  with _collapse_std(name=name, redirect_fn=_redirect_stdall):
    yield
