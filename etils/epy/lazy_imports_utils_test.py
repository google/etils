# Copyright 2024 The etils Authors.
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

"""Tests."""

import sys
from unittest import mock

from etils import ecolab
from etils import epy
import pytest

LazyModule = epy.lazy_imports_utils.LazyModule


@pytest.fixture(autouse=True)
def _clear_cache():
  # Reset the cache for each test
  ecolab.clear_cached_modules('dataclass_array')


def test_nested_import():
  assert 'dataclass_array' not in sys.modules
  with epy.lazy_imports():
    import dataclass_array.ops  # pylint: disable=g-import-not-at-top
  assert 'dataclass_array' not in sys.modules
  _ = dataclass_array.ops.stack
  assert 'dataclass_array' in sys.modules


def test_nested_import_as():
  assert 'dataclass_array' not in sys.modules
  with epy.lazy_imports():
    import dataclass_array.ops as new_ops  # pylint: disable=g-import-not-at-top
  assert 'dataclass_array' not in sys.modules
  _ = new_ops.stack
  assert 'dataclass_array' in sys.modules


def test_nested_import_from():
  assert 'dataclass_array' not in sys.modules
  with epy.lazy_imports():
    from dataclass_array import ops  # pylint: disable=g-import-not-at-top
  assert 'dataclass_array' not in sys.modules
  _ = ops.stack
  assert 'dataclass_array' in sys.modules


def test_import_with_alias():
  assert 'dataclass_array' not in sys.modules
  with epy.lazy_imports():
    import dataclass_array as dca  # pylint: disable=g-import-not-at-top
  assert 'dataclass_array' not in sys.modules
  _ = dca.stack
  assert 'dataclass_array' in sys.modules


def test_error_callback():
  success_callback = mock.MagicMock()
  error_callback = mock.MagicMock()
  with epy.lazy_imports(
      error_callback=error_callback, success_callback=success_callback
  ):
    import doesnotexist  # pylint: disable=g-import-not-at-top,unused-import # pytype: disable=import-error
  error_callback.assert_not_called()
  success_callback.assert_not_called()
  try:
    _ = doesnotexist.stack
  except ImportError:
    pass
  error_callback.assert_called_once()
  success_callback.assert_not_called()


def test_success_callback():
  success_callback = mock.MagicMock()
  error_callback = mock.MagicMock()
  with epy.lazy_imports(
      error_callback=error_callback, success_callback=success_callback
  ):
    from dataclass_array import ops  # pylint: disable=g-import-not-at-top
  error_callback.assert_not_called()
  success_callback.assert_not_called()
  _ = ops.stack
  error_callback.assert_not_called()
  success_callback.assert_called_once_with('dataclass_array.ops')
