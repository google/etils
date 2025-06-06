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

"""Tests pydantic serialization of etils.epath.Path."""

from unittest import mock

from etils import epath
import pydantic
import pytest


_GS_PREFIXES = ["gs"]


@pytest.mark.parametrize("gs_prefix", _GS_PREFIXES)
@pytest.mark.parametrize(
    "path_str",
    [
        "/example/path",
        "gs://example/path",
        "gs://example/path",
        "gs://example/path.ext",
        ".",
    ],
)
def test_pydantic_serialize(gs_prefix: str, path_str: str):

  class _Model(pydantic.BaseModel):
    path_field: epath.Path

  gs_schema = f"{gs_prefix}://"
  # Simulate being on XCloud/open-source, v.s. Borg.
  with mock.patch.dict(
      "etils.epath.gpath._URI_MAP_ROOT",
      {"gs://": f"/{gs_prefix}/"},
  ):
    path = epath.Path(path_str)
    # Outside of Borg, paths should be kept as URIs, i.e. gs://foo,
    # not /gcs/foo.
    if gs_prefix == "gs" and path_str.startswith(gs_schema):
      assert str(path).startswith(gs_schema)
    model = _Model(path_field=path)
    serialized = model.model_dump_json()
    deserialized = _Model.model_validate_json(serialized)
    assert isinstance(deserialized.path_field, epath.Path)
    assert model == deserialized
