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


def test_simple_import():
  assert 'dataclass_array' not in sys.modules
  error_callback = mock.Mock()
  success_callback = mock.Mock()
  with epy.lazy_imports(
      error_callback=error_callback,
      success_callback=success_callback,
  ):
    import dataclass_array  # pylint: disable=g-import-not-at-top
  error_callback.assert_not_called()
  success_callback.assert_not_called()
  assert 'dataclass_array' not in sys.modules
  _ = dataclass_array.stack
  assert 'dataclass_array' in sys.modules
  error_callback.assert_not_called()
  success_callback.assert_called_once()
  kwargs = success_callback.call_args.kwargs
  assert kwargs['module_name'] == 'dataclass_array'
  assert kwargs['import_time_ms'] > 0


def test_import_failure():
  error_callback = mock.Mock()
  success_callback = mock.Mock()
  with epy.lazy_imports(
      error_callback=error_callback,
      success_callback=success_callback,
  ):
    import thispackagedoesnotexist  # pylint: disable=g-import-not-at-top,unused-import  # pytype: disable=import-error
  with pytest.raises(ImportError):
    thispackagedoesnotexist.one_function()
  success_callback.assert_not_called()
  error_callback.assert_called_once()
  kwargs = error_callback.call_args.kwargs
  assert kwargs['module_name'] == 'thispackagedoesnotexist'
  assert isinstance(kwargs['exception'], ModuleNotFoundError)
