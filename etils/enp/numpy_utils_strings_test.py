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

"""String normalization tests without optional array frameworks."""

from etils import enp
import numpy as np
import pytest


@pytest.mark.parametrize('dtype', ['S8', 'U8', object])
@pytest.mark.parametrize('shape', [(0,), (0, 3), (2, 0, 3)])
def test_normalize_bytes2str_empty(dtype, shape):
  array = np.empty(shape, dtype=dtype)

  normalized = enp.normalize_bytes2str(array)

  assert normalized.shape == shape
  assert normalized.dtype.kind == 'U'
  assert normalized.size == 0
  assert array.dtype == np.dtype(dtype)


def test_normalize_bytes2str_filtered_batch():
  features = np.array([b'train', b'validation'], dtype=object)
  keep = np.array([False, False])

  normalized = enp.normalize_bytes2str(features[keep])

  assert normalized.shape == (0,)
  assert normalized.dtype.kind == 'U'
  np.testing.assert_array_equal(features, [b'train', b'validation'])


@pytest.mark.parametrize('dtype', ['S16', 'U16', object])
def test_normalize_bytes2str_unicode(dtype):
  text = [['a', 'caf\u00e9'], ['', '\u65e5\u672c\u8a9e']]
  values = (
      text
      if dtype == 'U16'
      else [[value.encode('utf8') for value in row] for row in text]
  )
  array = np.array(values, dtype=dtype)

  normalized = enp.normalize_bytes2str(array)

  assert normalized.shape == (2, 2)
  assert normalized.dtype.kind == 'U'
  np.testing.assert_array_equal(normalized, text)


@pytest.mark.parametrize('dtype', ['S8', 'U8', object])
def test_normalize_bytes2str_scalar_array(dtype):
  value = 'caf\u00e9' if dtype == 'U8' else 'caf\u00e9'.encode('utf8')

  normalized = enp.normalize_bytes2str(np.array(value, dtype=dtype))

  assert normalized.shape == ()
  assert normalized.item() == 'caf\u00e9'


def test_normalize_bytes2str_numeric_empty():
  array = np.empty((2, 0), dtype=np.float32)

  assert enp.normalize_bytes2str(array) is array


def test_normalize_bytes2str_invalid_utf8():
  with pytest.raises(UnicodeDecodeError):
    enp.normalize_bytes2str(np.array([b'\xff']))
