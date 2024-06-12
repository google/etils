# Copyright 2024 The etils Authors.
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

"""Tests pydantic serialization of etils.epath.Path."""

from etils import epath
import pydantic


def test_pydantic_serialize():

  class _Model(pydantic.BaseModel):
    path_field: epath.Path

  model = _Model(path_field=epath.Path("/example/path"))
  serialized = model.model_dump_json()
  deserialized = _Model.model_validate_json(serialized)
  assert isinstance(deserialized.path_field, epath.Path)
  assert model == deserialized
