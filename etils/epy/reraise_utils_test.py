# Copyright 2022 The etils Authors.
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

"""Tests for reraise_utils."""

import re

from etils import epy
import pytest


class CustomError(Exception):

  def __init__(self, *args, **kwargs):  # pylint: disable=super-init-not-called
    # Do not call super() to ensure this would work with bad code.
    self.custom_args = args
    self.custom_kwargs = kwargs


class CustomWithStrError(CustomError):

  def __str__(self) -> str:
    return f'custom error: {self.custom_args}, {self.custom_kwargs}'


def test_reraise():
  # No args
  with pytest.raises(ValueError, match='Caught: '):
    with epy.maybe_reraise(prefix='Caught: '):
      raise ValueError

  with pytest.raises(ValueError, match='Caught: With message'):
    with epy.maybe_reraise(prefix='Caught: '):
      raise ValueError('With message')

  # Custom exception
  with pytest.raises(CustomError, match='123\nCaught!') as exc_info:
    with epy.maybe_reraise(suffix='Caught!'):
      raise CustomError(123, y=345)
  assert str(exc_info.value) == '123\nCaught!'
  assert repr(exc_info.value) == "CustomError('123\\nCaught!')"
  assert exc_info.value.custom_args == (123,)
  assert exc_info.value.custom_kwargs == {'y': 345}

  # Lazy-message
  with pytest.raises(CustomError, match=re.escape('Caught: (123, {})')):
    with epy.maybe_reraise(prefix=lambda: 'Caught: '):
      raise CustomError(123, {})

  with pytest.raises(CustomError, match=re.escape("Caught: ('a', 'b', 'c')")):
    with epy.maybe_reraise(prefix='Caught: '):
      ex = CustomError(123, {})
      ex.args = 'abc'  # Not a tuple
      raise ex

  with pytest.raises(ImportError, match='Caught: With message'):
    with epy.maybe_reraise(prefix='Caught: '):
      raise ImportError('With message', name='xyz')

  with pytest.raises(FileNotFoundError, match='Caught: With message'):
    with epy.maybe_reraise(prefix='Caught: '):
      raise FileNotFoundError('With message')


def test_no_cause():
  with pytest.raises(CustomWithStrError) as exc_info:
    with epy.maybe_reraise(prefix='Caught2: '):
      with epy.maybe_reraise(prefix='Caught: '):
        raise CustomWithStrError(None, 'message', x=2)

  e = exc_info.value
  assert isinstance(e, CustomWithStrError)
  assert (
      "Caught2: Caught: custom error: (None, 'message'), {'x': 2}") == str(e)
  assert ('CustomWithStrError("Caught2: Caught: custom error: (None, '
          '\'message\'), {\'x\': 2}")') == repr(e)
  # Attributes are correctly forwarded
  assert e.custom_args == (None, 'message')
  assert e.custom_kwargs == {'x': 2}
  # Only a single cause is set (so avoid nested context)
  assert e.__cause__ is None
  assert e.__suppress_context__


def test_with_cause():
  with pytest.raises(ImportError) as exc_info:
    with epy.maybe_reraise(prefix='Caught2: '):
      with epy.maybe_reraise(prefix='Caught: '):
        try:
          e_origin = ValueError()
          raise e_origin
        except ValueError as e:
          raise ImportError from e

  e = exc_info.value
  assert 'Caught2: Caught: ' == str(e)
  assert "ImportError('Caught2: Caught: ')" == repr(e)
  # The original cause is properly forwarded (and displayed)
  assert e.__cause__ is e_origin
  assert e.__suppress_context__
