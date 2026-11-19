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

"""Import-scope isolation and restoration tests."""

import builtins
import contextlib
import gc
import json
import threading
import weakref

from etils import epy
from etils.epy import lazy_imports_utils
import pytest


@pytest.fixture(autouse=True)
def _restore_import():
  # The restoration regression must not leave a broken hook in the test runner.
  original = builtins.__import__
  try:
    yield
  finally:
    builtins.__import__ = original


def _plain_import():
  from json import dumps  # pylint: disable=g-import-not-at-top,g-importing-member

  return dumps


def _relative_import():
  from . import env_utils  # pylint: disable=g-import-not-at-top

  return env_utils


def test_called_function_import_is_eager():
  with epy.lazy_imports():
    dumps = _plain_import()
  assert dumps is json.dumps
  assert dumps({'ok': True}) == '{"ok": true}'


def test_called_function_relative_import_is_eager():
  with epy.lazy_imports():
    module = _relative_import()
  assert module.__name__ == 'etils.epy.env_utils'


def test_called_function_keyword_import_is_eager():
  def import_with_keywords():
    return __import__(
        'json', globals=globals(), locals=locals(), fromlist=('dumps',), level=0
    ).dumps

  with epy.lazy_imports():
    dumps = import_with_keywords()
  assert dumps is json.dumps


def test_direct_keyword_import_is_lazy():
  with epy.lazy_imports():
    module = __import__(
        'missing_scope_test_module',
        globals=globals(),
        locals=locals(),
        fromlist=(),
        level=0,
    )
  assert isinstance(module, lazy_imports_utils.LazyModule)


@pytest.mark.parametrize('import_fn', [_plain_import, _relative_import])
def test_other_thread_import_is_eager(import_fn):
  ready = threading.Event()
  finished = threading.Event()
  results = []

  def worker():
    try:
      if not ready.wait(10):
        raise TimeoutError('Lazy-import scope did not start')
      results.append(import_fn())
    except BaseException as error:  # Capture for assertion on the main thread.
      results.append(error)
    finally:
      finished.set()

  thread = threading.Thread(target=worker)
  thread.start()
  try:
    with epy.lazy_imports():
      ready.set()
      assert finished.wait(10)
  finally:
    ready.set()
    thread.join(10)
  assert not thread.is_alive()
  assert results == [import_fn()]


@pytest.mark.parametrize('import_fn', [_plain_import, _relative_import])
def test_gc_finalizer_import_is_eager(import_fn):
  class Node:
    pass

  results = []

  def finalize():
    try:
      results.append(import_fn())
    except BaseException as error:
      results.append(error)

  node = Node()
  node.cycle = node
  finalizer = weakref.finalize(node, finalize)
  del node
  with epy.lazy_imports():
    gc.collect()
  assert not finalizer.alive
  assert results == [import_fn()]


@pytest.mark.parametrize('exit_first_entered_first', [False, True])
def test_overlapping_thread_scopes(exit_first_entered_first):
  original = builtins.__import__
  entered = [threading.Event(), threading.Event()]
  release = [threading.Event(), threading.Event()]
  exited = [threading.Event(), threading.Event()]
  results = [[], []]
  callbacks = [[], []]
  errors = []

  def worker(index):
    try:
      with epy.lazy_imports(success_callback=callbacks[index].append):
        import json as first  # pylint: disable=g-import-not-at-top

        results[index].append(first)
        entered[index].set()
        if not release[index].wait(10):
          raise TimeoutError('Lazy-import scope was not released')
        import json as second  # pylint: disable=g-import-not-at-top

        results[index].append(second)
    except BaseException as error:
      errors.append(error)
    finally:
      exited[index].set()

  threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
  try:
    threads[0].start()
    assert entered[0].wait(10)
    threads[1].start()
    assert entered[1].wait(10)
    first_exit = 0 if exit_first_entered_first else 1
    release[first_exit].set()
    assert exited[first_exit].wait(10)
    release[1 - first_exit].set()
  finally:
    for event in release:
      event.set()
    for thread in threads:
      if thread.ident is not None:
        thread.join(10)

  assert not any(thread.is_alive() for thread in threads)
  restored = builtins.__import__ is original
  # Restore before any assertion, even when running the test on broken main.
  builtins.__import__ = original
  assert not errors
  assert restored
  for index in range(2):
    assert len(results[index]) == 2
    for module in results[index]:
      assert isinstance(module, lazy_imports_utils.LazyModule)
      assert module.dumps({'ok': True}) == '{"ok": true}'
    assert callbacks[index] == ['json', 'json']


def test_nested_scope_callbacks():
  outer_calls, inner_calls = [], []
  original = builtins.__import__
  with epy.lazy_imports(success_callback=outer_calls.append):
    import json as outer_before  # pylint: disable=g-import-not-at-top

    with epy.lazy_imports(success_callback=inner_calls.append):
      import json as inner  # pylint: disable=g-import-not-at-top

    import json as outer_after  # pylint: disable=g-import-not-at-top

  assert builtins.__import__ is original
  assert not outer_calls and not inner_calls
  assert outer_before.dumps([]) == inner.dumps([]) == outer_after.dumps([])
  assert outer_calls == ['json', 'json']
  assert inner_calls == ['json']


@pytest.mark.parametrize('stack_cls', [contextlib.ExitStack, epy.ExitStack])
def test_exit_stack_preserves_lazy_import(stack_cls):
  if stack_cls is epy.ExitStack:
    stack = stack_cls([epy.lazy_imports()])
  else:
    stack = stack_cls()
  with stack:
    if stack_cls is contextlib.ExitStack:
      stack.enter_context(epy.lazy_imports())
    import missing_scope_test_module as module  # pylint: disable=g-import-not-at-top,import-error

  assert isinstance(module, lazy_imports_utils.LazyModule)


def test_delegating_context_preserves_lazy_import():
  @contextlib.contextmanager
  def delegate():
    with epy.lazy_imports():
      yield

  with delegate():
    import missing_scope_test_module as module  # pylint: disable=g-import-not-at-top,import-error

  assert isinstance(module, lazy_imports_utils.LazyModule)


def test_context_decorator_preserves_lazy_import():
  @epy.lazy_imports()
  def decorated():
    import missing_scope_test_module as module  # pylint: disable=g-import-not-at-top,import-error

    return module

  assert isinstance(decorated(), lazy_imports_utils.LazyModule)


def test_callable_context_decorator_preserves_lazy_import():
  class Callable:

    def __call__(self):
      import missing_scope_test_module as module  # pylint: disable=g-import-not-at-top,import-error

      return module

  decorated = epy.lazy_imports()(Callable())  # pylint: disable=not-callable
  assert isinstance(decorated(), lazy_imports_utils.LazyModule)


def test_class_context_preserves_lazy_import():
  class Delegate:

    def __enter__(self):
      self.context = epy.lazy_imports()
      return self.context.__enter__()

    def __exit__(self, *args):
      return self.context.__exit__(*args)

  with Delegate():
    import missing_scope_test_module as module  # pylint: disable=g-import-not-at-top,import-error

  assert isinstance(module, lazy_imports_utils.LazyModule)


def test_existing_import_hook_is_preserved(monkeypatch):
  original = builtins.__import__
  imported = []

  def tracking_import(*args, **kwargs):
    imported.append(args[0])
    return original(*args, **kwargs)

  monkeypatch.setattr(builtins, '__import__', tracking_import)
  with epy.lazy_imports():
    import missing_scope_test_module as module  # pylint: disable=g-import-not-at-top,import-error

    dumps = _plain_import()
  assert builtins.__import__ is tracking_import
  assert isinstance(module, lazy_imports_utils.LazyModule)
  assert dumps is json.dumps
  assert 'json' in imported
  assert 'missing_scope_test_module' not in imported


def test_scope_does_not_retain_caller_frame():
  class Node:
    pass

  def owner():
    node = Node()
    ref = weakref.ref(node)
    with epy.lazy_imports():
      import missing_scope_test_module as module  # pylint: disable=g-import-not-at-top,import-error

    assert isinstance(module, lazy_imports_utils.LazyModule)
    return ref

  ref = owner()
  gc.collect()
  assert ref() is None


def test_direct_relative_import_still_rejected():
  with pytest.raises(
      ValueError, match='Relative import statements not supported'
  ):
    with epy.lazy_imports():
      from . import env_utils  # pylint: disable=g-import-not-at-top,unused-import


def test_exception_restores_import():
  original = builtins.__import__
  with pytest.raises(ValueError, match='scope failure'):
    with epy.lazy_imports():
      raise ValueError('scope failure')
  assert builtins.__import__ is original
  assert _plain_import() is json.dumps
