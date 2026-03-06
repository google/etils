# Copyright 2026 The etils Authors.
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

"""Tests."""

import concurrent.futures
import dataclasses
import threading
from typing import List

from etils import edc


@edc.dataclass
@dataclasses.dataclass(frozen=True)
class Context:
  thread_id: edc.ContextVar[int] = dataclasses.field(
      default_factory=threading.get_native_id
  )
  stack: edc.ContextVar[List[int]] = dataclasses.field(default_factory=list)


def test_context():
  # Global context object
  context = Context(thread_id=0)
  assert context.thread_id == 0
  assert context.stack == []  # pylint: disable=g-explicit-bool-comparison
  context.stack.append(0)
  context.stack.append(1)
  assert context.stack == [0, 1]

  def worker():
    # Inside each thread, the worker use it's own context
    assert context.thread_id != 0
    assert context.stack == []  # pylint: disable=g-explicit-bool-comparison
    context.stack.append(1)

  with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    for _ in range(10):
      executor.submit(worker)


def test_default_factory_thread_isolation():
  ctx = Context()

  ctx.stack.append(99)

  errors = []
  barrier = threading.Barrier(4)

  def worker():
    try:
      barrier.wait(timeout=5)
      local_stack = ctx.stack
      assert not local_stack, f'Expected empty list, got {local_stack!r}'
      local_stack.append(1)
      assert len(local_stack) == 1, f'Expected length 1, got {len(local_stack)}'
    except Exception as e:  # pylint: disable=broad-except
      errors.append(e)

  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    for _ in range(4):
      executor.submit(worker)

  assert not errors, f'Thread isolation violated: {errors}'
