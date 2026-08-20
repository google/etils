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

"""Wrapper for tdqm."""

import logging as logging_stdlib
import sys
import typing
from typing import Optional, TypeVar

from etils import epy
from etils.epy import _internal

with _internal.check_missing_deps():
  # pylint: disable=g-import-not-at-top
  from absl import logging
  import tqdm as tqdm_base
  # pylint: enable=g-import-not-at-top

_IterableT = TypeVar('_IterableT')


class _LogFile:
  """A tqdm-compatible file-like object that logs to INFO.

  Captures the caller's source location at construction time so that
  log messages attribute to the code that created the progress bar,
  not to tqdm internals.
  """

  def __init__(self, caller_depth: int = 2) -> None:
    """Initializes the log file.

    Args:
      caller_depth: Number of frames to skip above ``_LogFile.__init__`` to
        reach the "real" caller.  Default is 2, which skips ``__init__`` itself
        and one wrapper (e.g. ``tqdm()``).
    """
    frame = sys._getframe(caller_depth)
    self._caller_file = frame.f_code.co_filename
    self._caller_lineno = frame.f_lineno
    self._caller_func = frame.f_code.co_name

  def write(self, message: str) -> None:
    """Logs a non-empty message at INFO level with the captured source location."""
    if message := message.strip():
      logger = logging.get_absl_logger()
      record = logger.makeRecord(
          name=logger.name,
          level=logging_stdlib.INFO,
          fn=self._caller_file,
          lno=self._caller_lineno,
          msg=message,
          args=(),
          exc_info=None,
          func=self._caller_func,
      )
      logger.handle(record)

  def flush(self) -> None:
    pass

  def close(self) -> None:
    pass


# TODO(epot): Mock the original `tqdm`, (in `__main__`), rather than
# having to change the import.
def tqdm(iterable: Optional[_IterableT] = None, **kwargs) -> _IterableT:
  """Add a progressbar to the iterable."""
  return tqdm_base.tqdm(iterable=iterable, **kwargs)  # pyrefly: ignore[bad-return]


if typing.TYPE_CHECKING:
  # API is the same as open-source
  tqdm = tqdm_base.tqdm
