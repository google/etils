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

"""Colab public API."""

from etils.ecolab.array_as_img import auto_plot_array
from etils.ecolab.colab_utils import collapse
from etils.ecolab.module_utils import clear_cached_modules
from etils.ecolab.patch_utils import patch_graphviz
from etils.ecolab.patch_utils import set_verbose

__all__ = [
    'auto_plot_array',
    'collapse',
    'clear_cached_modules',
    'patch_graphviz',
    'set_verbose',
]
