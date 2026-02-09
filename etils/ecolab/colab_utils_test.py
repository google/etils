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

"""Tests for ecolab."""

import sys

from etils import ecolab
import pytest


# IPython widget do not work in unittests
@pytest.mark.skip
def test_capture_stdout():
  with ecolab.collapse():
    print('Abcd')
    print('Abcd', file=sys.stderr)


def test_json():
  ecolab.json({'a': [1, None, 3.4]})
  ecolab.json([1, None, 3.4, {'d': [1, True]}])


def test_get_permalink():
  assert (
      ecolab.get_permalink(
          url='https://colab.research.google.com/some_file.ipynb',
          template_params={
              'A': 1,
              'B': 'b',
          },
      )
      == 'https://colab.research.google.com/some_file.ipynb#templateParams=%7B%22A%22%3A%201%2C%20%22B%22%3A%20%22b%22%7D'
  )
  assert (
      ecolab.get_permalink(
          url='https://colab.research.google.com/some_file.ipynb',
          template_params=(
              ('A', 1, 1),
              ('B', 'b', ''),
          ),
      )
      == 'https://colab.research.google.com/some_file.ipynb#templateParams=%7B%22B%22%3A%20%22b%22%7D'
  )
