# Copyright 2025 The etils Authors.
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


@pytest.fixture(autouse=True)
def _clear_cache():
  # Reset the cache for each test
  ecolab.clear_cached_modules('tensorflow_datasets')


def test_nested_import():
  assert 'tensorflow_datasets' not in sys.modules
  with epy.lazy_imports():
    import tensorflow_datasets.core  # pylint: disable=g-import-not-at-top
  assert 'tensorflow_datasets' not in sys.modules
  _ = tensorflow_datasets.core.features
  assert 'tensorflow_datasets' in sys.modules


def test_nested_import_as():
  assert 'tensorflow_datasets' not in sys.modules
  with epy.lazy_imports():
    import tensorflow_datasets.core as new_core  # pylint: disable=g-import-not-at-top
  assert 'tensorflow_datasets' not in sys.modules
  _ = new_core.features
  assert 'tensorflow_datasets' in sys.modules


def test_nested_import_from():
  assert 'tensorflow_datasets' not in sys.modules
  with epy.lazy_imports():
    from tensorflow_datasets import core  # pylint: disable=g-import-not-at-top
  assert 'tensorflow_datasets' not in sys.modules
  _ = core.features
  assert 'tensorflow_datasets' in sys.modules


def test_import_with_alias():
  assert 'tensorflow_datasets' not in sys.modules
  with epy.lazy_imports():
    import tensorflow_datasets as tfds  # pylint: disable=g-import-not-at-top
  assert 'tensorflow_datasets' not in sys.modules
  _ = tfds.features
  assert 'tensorflow_datasets' in sys.modules


def test_setattr():
  assert 'tensorflow_datasets' not in sys.modules
  with epy.lazy_imports():
    import tensorflow_datasets as tfds  # pylint: disable=g-import-not-at-top
  assert 'tensorflow_datasets' not in sys.modules
  tfds.features = 'foo'
  assert 'tensorflow_datasets' in sys.modules
  assert tfds.features == 'foo'


def test_delattr():
  assert 'tensorflow_datasets' not in sys.modules
  with epy.lazy_imports():
    import tensorflow_datasets as tfds  # pylint: disable=g-import-not-at-top
  assert 'tensorflow_datasets' not in sys.modules
  del tfds.features
  assert 'tensorflow_datasets' in sys.modules
  with pytest.raises(AttributeError):
    _ = tfds.features


def test_delattr_submodule():
  assert 'tensorflow_datasets' not in sys.modules
  with epy.lazy_imports():
    import tensorflow_datasets.core  # pylint: disable=g-import-not-at-top
  assert 'tensorflow_datasets' not in sys.modules
  del tensorflow_datasets.core
  with pytest.raises(AttributeError):
    _ = tensorflow_datasets.core


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
    _ = doesnotexist.features
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
    from tensorflow_datasets import core  # pylint: disable=g-import-not-at-top
  error_callback.assert_not_called()
  success_callback.assert_not_called()
  _ = core.features
  error_callback.assert_not_called()
  success_callback.assert_called_once_with('tensorflow_datasets.core')


def test_import_fail():
  assert 'tensorflow_datasets' not in sys.modules
  with mock.patch(
      'importlib.import_module', side_effect=AttributeError('no attribute')
  ):
    with epy.lazy_imports():
      import tensorflow_datasets as tfds  # pylint: disable=g-import-not-at-top

    with pytest.raises(ImportError, match='no attribute'):
      _ = tfds.features
