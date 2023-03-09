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

"""Torch compatibility."""

from __future__ import annotations

from etils.enp import numpy_utils

lazy = numpy_utils.lazy


def activate_torch_support() -> None:
  """Activate numpy behavior for `torch`.

  This function mocks `torch` to make its behavior closer to `numpy` by:

  *   Adding some missing methods (`torch.ndarray`, `torch.expand_dims`,...)
  *   Mocking a few methods to support `np.dtype` (
      https://github.com/pytorch/pytorch/issues/40568)
  """
  torch = lazy.torch
  if hasattr(torch, '__etils_np_mode__'):  # Already mocked
    return
  torch.__etils_np_mode__ = True

  # 'tensor',
  # 'asarray',
  # 'zeros',
  # 'ones',
