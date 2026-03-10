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
  # Global context object — shared across all threads.
  context = Context(thread_id=0)
  assert context.thread_id == 0
  assert context.stack == []  # pylint: disable=g-explicit-bool-comparison
  context.stack.append(0)
  context.stack.append(1)
  assert context.stack == [0, 1]

  # Use a barrier, so each worker is launched in a separate thread. Otherwise,
  # ThreadPoolExecutor reuses threads.
  barrier = threading.Barrier(5)

  def worker():
    barrier.wait(timeout=5)
    assert context.thread_id == threading.get_native_id()
    assert context.stack == []  # pylint: disable=g-explicit-bool-comparison
    context.stack.append(1)
    assert context.stack == [1]

  with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(worker) for _ in range(5)]
    for f in futures:
      f.result()  # Re-raises worker exceptions in the main thread


def test_context_default_thread_id():
  context = Context()
  assert context.thread_id == threading.get_native_id()
  assert context.stack == []  # pylint: disable=g-explicit-bool-comparison
  context.stack.append(0)
  context.stack.append(1)
  assert context.stack == [0, 1]

  # Use a barrier, so each worker is launched in a separate thread. Otherwise,
  # ThreadPoolExecutor reuses threads.
  barrier = threading.Barrier(5)

  def worker():
    barrier.wait(timeout=5)
    # Each worker must see its own native thread ID, not the main thread's.
    assert context.thread_id == threading.get_native_id()
    assert context.stack == []  # pylint: disable=g-explicit-bool-comparison
    context.stack.append(1)
    assert context.stack == [1]

  with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(worker) for _ in range(5)]
    for f in futures:
      f.result()
