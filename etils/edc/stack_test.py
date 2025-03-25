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

import concurrent.futures

from etils import edc


def test_stack():
  # Global context object
  stack = edc.ContextStack()

  assert not stack

  def worker():
    # Inside each thread, the worker use it's own context
    assert not stack
    assert len(stack) == 0  # pylint: disable=g-explicit-bool-comparison,g-explicit-length-test
    assert stack.stack == []  # pylint: disable=g-explicit-bool-comparison
    stack.append(1)
    stack.append(4)
    assert stack
    assert len(stack) == 2
    assert stack[-1] == 4
    assert stack[-2] == 1
    assert stack.pop() == 4
    assert stack.pop() == 1
    assert not stack
    stack.append(1)
    stack.append(4)
    assert stack

  with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    for _ in range(10):
      executor.submit(worker)

  # The original stack is always empty, even though the workers have added
  # values.
  assert not stack
