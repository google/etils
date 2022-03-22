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

"""Tests for etils.ecolab.array_as_img."""

from __future__ import annotations

from unittest import mock

from etils import enp
from etils.ecolab import array_as_img
import pytest


# Skip the test because it require a more recent mediapy version.
# TODO(epot): Restore tests
pytest.skip(allow_module_level=True)


def test_display_array_as_image():
  with mock.patch('IPython.get_ipython', mock.MagicMock()) as ipython_mock:
    array_as_img.auto_plot_array()
  assert ipython_mock.call_count == 1


@pytest.mark.parametrize('valid_shape', [
    (28, 28),
    (28, 28, 1),
    (28, 28, 3),
    (28, 28, 4),
    (4, 28, 28, 1),
    (4, 28, 28, 3),
    (4, 28, 28, 4),
    (1, 28, 28, 3),
])
@enp.testing.parametrize_xnp()
def test_array_repr_html_valid(
    xnp: enp.NpModule,
    valid_shape: tuple[int, ...],
):
  # 2D images are displayed as images
  assert '<img' in array_as_img._array_repr_html(
      xnp.zeros(valid_shape),
      video_min_num_frames=15,
  )


@pytest.mark.parametrize('valid_shape', [
    (20, 28, 28, 3),
])
@enp.testing.parametrize_xnp()
def test_array_repr_video_html_valid(
    xnp: enp.NpModule,
    valid_shape: tuple[int, ...],
):
  # 2D images are displayed as video
  assert '<video' in array_as_img._array_repr_html(
      xnp.zeros(valid_shape),
      video_min_num_frames=15,
  )


@pytest.mark.parametrize(
    'invalid_shape',
    [
        (7, 7),
        (28, 7),  # Only one dimension bigger than the threshold
        (28, 28, 5),  # Invalid number of dimension
        (28, 28, 0),
        (2, 28, 28),
        (2, 28, 28, 5),
        (0, 28, 28, 3),
        (20, 28, 28, 1),
        (20, 28, 28, 5),
    ],
)
@enp.testing.parametrize_xnp()
def test_array_repr_html_invalid(
    xnp: enp.NpModule,
    invalid_shape: tuple[int, ...],
):
  assert array_as_img._array_repr_html(
      xnp.zeros(invalid_shape),
      video_min_num_frames=15,
  ) is None
