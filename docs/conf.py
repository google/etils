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

"""Generate documentation.

Usage (from the root directory):

```sh
pip install -e .[docs]

sphinx-build -b html docs/ docs/_build
```
"""

import os
import pathlib
import sys

# `apitree` depend on `etils` so add the local `etils` to `sys.module` before
# importing.
repo_dir = pathlib.Path(__file__).parent.parent  # <repo>/docs/conf.py
sys.path.insert(0, os.fspath(repo_dir))

import apitree  # pylint: disable=g-import-not-at-top


modules = {
    'array_types': 'etils.array_types',
    'eapp': 'etils.eapp',
    'ecolab': 'etils.ecolab',
    'edc': 'etils.edc',
    'enp': 'etils.enp',
    'epath': 'etils.epath',
    'epy': 'etils.epy',
    'etree': 'etils.etree',
    'lazy_imports': 'etils.lazy_imports',
}

apitree.make_project(
    modules=modules,
    includes_paths={
        f'etils/{alias}/README.md': f'{alias}.md' for alias in modules
    },
    globals=globals(),
)
