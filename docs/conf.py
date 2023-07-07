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

"""Documentation."""

import pathlib

import apitree
from etils import epy

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

# Could hide this in an event
# https://www.sphinx-doc.org/en/master/extdev/appapi.html#sphinx-core-events

root_dir = pathlib.Path(__file__).parent

for alias in modules:
  if alias == 'lazy_imports':
    continue  # No README.md
  content = epy.dedent(
      f"""
      ```{{include}} ../etils/{alias}/README.md
      ```
      """
  )
  root_dir.joinpath(f'{alias}.md').write_text(content)


apitree.make_project(
    modules=modules,
    globals=globals(),
)
