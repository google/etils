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

"""Tests for logging_utils."""

import io
import logging
import sys

from absl import logging as absl_logging
from absl.testing import flagsaver
from etils.eapp import logging_utils


def test_better_logging_emits_each_record_once(monkeypatch):
  root_logger = logging.getLogger()
  original_handlers = root_logger.handlers[:]
  original_level = root_logger.level
  absl_handler = absl_logging.get_absl_handler()
  python_handler = absl_handler.python_handler
  original_formatter = python_handler.formatter
  original_stream = python_handler.stream
  stream = io.StringIO()

  flags_were_parsed = logging_utils.FLAGS.is_parsed()
  if not flags_were_parsed:
    logging_utils.FLAGS(['logging_utils_test'])
  monkeypatch.delitem(sys.modules, 'tqdm', raising=False)

  try:
    with flagsaver.flagsaver(
        logtostderr=False,
        alsologtostderr=False,
        stderrthreshold='fatal',
    ):
      root_logger.handlers = [absl_handler]
      root_logger.setLevel(logging.INFO)
      python_handler.setStream(stream)

      logging_utils._better_logging()
      logging_utils._better_logging()
      root_logger.info('one record')

      assert absl_handler not in root_logger.handlers
      assert root_logger.handlers.count(python_handler) == 1
      assert stream.getvalue().count('one record') == 1
  finally:
    root_logger.handlers = original_handlers
    root_logger.setLevel(original_level)
    python_handler.setFormatter(original_formatter)
    python_handler.setStream(original_stream)
    if not flags_were_parsed:
      logging_utils.FLAGS.unparse_flags()
