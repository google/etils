# Copyright 2021 The etils Authors.
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

"""Compate the python version.

Outputs:

* `should-release`: Whether local_version > pypi_version ("true" or "false")

"""

from __future__ import annotations

import argparse

import packaging.version


def parse_arguments() -> argparse.Namespace:
  """Parse CLI arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--local_version",
      required=True,
      help="Github action output name.",
  )
  parser.add_argument(
      "--pypi_version",
      required=True,
      help="Github action output name.",
  )
  return parser.parse_args()


def main():
  args = parse_arguments()

  local_version = packaging.version.Version(args.local_version)
  pypi_version = packaging.version.Version(args.pypi_version)

  if local_version > pypi_version:
    should_release = "true"
  else:
    should_release = "false"
  print(f"::set-output name=should-release::{should_release}")


if __name__ == "__main__":
  main()
