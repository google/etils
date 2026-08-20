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

import logging
import logging.handlers
import os
import unittest
from unittest import mock

from etils import etqdm
from etils.etqdm import tqdm_utils


class TqdmBasicTest(unittest.TestCase):
  """Baseline test that tqdm wraps iterables correctly."""

  def test_tqdm_iterates(self):
    self.assertEqual(list(etqdm.tqdm(range(3))), [0, 1, 2])


if __name__ == '__main__':
  unittest.main()
