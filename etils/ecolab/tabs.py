# Copyright 2026 The etils Authors.
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

"""Tabs iterator."""

from collections.abc import Iterable, Iterator
from typing import TypeVar

import IPython.display
import ipywidgets as widgets

_T = TypeVar("_T")


def iter_tab(
    iterable: Iterable[_T],
    *,
    prefix: str | None = None,
) -> Iterator[_T]:
  """Displays the items of an iterable each in a numbered tab.

  Example:

  ```python
  for ex in ecolab.iter_tab(ds[:5]):
    print(ex)  # Each example is displayed in a separate tab.
  ```

  Args:
    iterable: The iterable to iterate over.
    prefix: The prefix to use for the tab titles.

  Yields:
    The items from the iterable.
  """
  tab = widgets.Tab()

  IPython.display.display(tab)

  # List to hold the child widgets (tabs)
  children = []

  for i, item in enumerate(iterable):
    out = widgets.Output()
    children.append(out)
    tab.children = tuple(children)

    tab_title = f"{prefix} {i}" if prefix else str(i)
    tab.set_title(i, tab_title)

    # Captures all stdout/stderr/display within the user's loop
    with out:
      yield item
