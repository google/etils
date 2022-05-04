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

"""Numpy API.

When possible, utils are meant to work with
both numpy and jax.numpy.

"""

import sys

from etils.enp import compat
from etils.enp.array_spec import ArraySpec
from etils.enp.interp_utils import interp
from etils.enp.numpy_utils import get_np_module
from etils.enp.numpy_utils import is_array
from etils.enp.numpy_utils import is_array_str
from etils.enp.numpy_utils import is_dtype_str
from etils.enp.numpy_utils import lazy
from etils.enp.numpy_utils import normalize_bytes2str
from etils.enp.numpy_utils import NpModule
from etils.enp.numpy_utils import tau

# TODO(epot): Deprecate compat and use `linalg` everywhere ?
linalg = compat

# Inside tests, can use `enp.testing`
if 'pytest' in sys.modules:  # < Ensure open source does not trigger import
  try:
    from etils.enp import testing  # pylint: disable=g-import-not-at-top
  except ImportError:
    pass

del sys
