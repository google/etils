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

"""`etils` internal utils."""

import contextlib
from typing import Iterator

from etils.epy import reraise_utils


@contextlib.contextmanager
def check_missing_deps() -> Iterator[None]:
  """Raise a better error message in case of `ImportError`.

  Usage:

  ```python
  from etils.epy import _internal

  with _internal.check_missing_deps():
    # pylint: disable=g-import-not-at-top
    import xyz
    # pylint: enable=g-import-not-at-top
  ```

  Yields:
    None
  """
  try:
    yield
  except ImportError as e:
    reraise_utils.reraise(
        e,
        suffix=(
            '\nEach etils sub-modules require deps to be installed separately '
            '(e.g. `from etils import ecolab` -> `pip install etils[ecolab]`)'
        ),
    )
