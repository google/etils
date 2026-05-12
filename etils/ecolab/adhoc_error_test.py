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

"""Tests for the leaking module detection in adhoc_error."""

# pylint: disable=redefined-outer-name
from __future__ import annotations

import sys
import types
from typing import Any

from etils.ecolab import adhoc_error
import pytest


def _make_module(name: str, attrs: dict[str, Any]) -> types.ModuleType:
  """Create a module with the given name and attributes."""
  module = types.ModuleType(name)
  module.__dict__.update(attrs)
  sys.modules[name] = module
  return module


def _invalidate_module(module: types.ModuleType) -> None:
  """Simulate the invalidation done by adhoc reload."""
  module_name = module.__name__
  module.__dict__.clear()
  module.__etils_invalidated__ = True
  module.__dict__['__name__'] = module_name


@pytest.fixture
def cleanup_modules():
  """Remove test modules from sys.modules after the test."""
  created = []
  yield created
  for name in created:
    sys.modules.pop(name, None)


class TestGetInvalidatedModuleNames:
  """Tests for _get_invalidated_module_names."""

  def test_no_invalidated_module(self):
    try:
      raise NameError("name 'x' is not defined")
    except NameError as e:
      result = adhoc_error._get_invalidated_module_names(e)
    assert result == set()

  def test_with_invalidated_module(self, cleanup_modules):
    module = _make_module('test_invalidated_mod', {'x': 1})
    cleanup_modules.append('test_invalidated_mod')

    # Simulate code running in the module's globals then getting invalidated
    code = compile("raise NameError('x')", 'test_invalidated_mod.py', 'exec')
    try:
      exec(code, module.__dict__)  # pylint: disable=exec-used
    except NameError:
      pass

    # Now invalidate the module
    _invalidate_module(module)

    # Create an exception that would come from the invalidated module
    # We need a traceback frame whose f_globals contains __etils_invalidated__
    # The simplest way is to execute code in the invalidated module's namespace
    result = set()
    try:
      exec(  # pylint: disable=exec-used
          compile("raise NameError('MISSING')", 'test.py', 'exec'),
          module.__dict__,
      )
    except NameError as e:
      result = adhoc_error._get_invalidated_module_names(e)

    assert result == {'test_invalidated_mod'}


class TestHasInvalidatedGlobals:
  """Tests for _has_invalidated_globals."""

  def test_function_with_invalidated_globals(self):
    """A function whose __globals__ was invalidated should be detected."""
    invalidated_globals = {'__etils_invalidated__': True}

    func = types.FunctionType(
        compile('pass', '<test>', 'exec'),
        invalidated_globals,
        'stale_func',
    )
    assert adhoc_error._has_invalidated_globals(func)

  def test_function_with_clean_globals(self):
    """A function with normal globals should not be detected."""
    func = types.FunctionType(
        compile('pass', '<test>', 'exec'),
        {'some_var': 1},
        'clean_func',
    )
    assert not adhoc_error._has_invalidated_globals(func)

  def test_class_with_invalidated_method(self):
    """A class whose method has invalidated globals should be detected."""
    invalidated_globals = {'__etils_invalidated__': True}

    stale_method = types.FunctionType(
        compile('pass', '<test>', 'exec'),
        invalidated_globals,
        'method',
    )

    MyClass = type('MyClass', (), {'method': stale_method})
    assert adhoc_error._has_invalidated_globals(MyClass)

  def test_class_with_invalidated_metaclass_method(self):
    """A class whose metaclass method has invalidated globals."""
    invalidated_globals = {'__etils_invalidated__': True}

    stale_getitem = types.FunctionType(
        compile('pass', '<test>', 'exec'),
        invalidated_globals,
        '__getitem__',
    )

    Meta = type('Meta', (type,), {'__getitem__': stale_getitem})
    MyClass = Meta('MyClass', (), {})
    assert adhoc_error._has_invalidated_globals(MyClass)

  def test_class_with_clean_metaclass(self):
    """A class with a normal metaclass should not be detected."""
    MyClass = type('MyClass', (), {})
    assert not adhoc_error._has_invalidated_globals(MyClass)

  def test_plain_value(self):
    """Non-callable values should not be detected."""
    assert not adhoc_error._has_invalidated_globals(42)
    assert not adhoc_error._has_invalidated_globals('hello')


class TestFindLeakingModules:
  """Tests for _find_leaking_modules."""

  def test_no_leakers(self):
    assert not adhoc_error._find_leaking_modules()

  def test_finds_module_with_stale_function(self, cleanup_modules):
    invalidated_globals = {'__etils_invalidated__': True}

    stale_func = types.FunctionType(
        compile('pass', '<test>', 'exec'),
        invalidated_globals,
        'stale_func',
    )

    _make_module('leaker_mod', {'stale_func': stale_func})
    cleanup_modules.append('leaker_mod')

    leakers = adhoc_error._find_leaking_modules()
    assert 'leaker_mod' in leakers

  def test_finds_module_with_stale_metaclass(self, cleanup_modules):
    """The real-world scenario: class with metaclass from invalidated module."""
    invalidated_globals = {'__etils_invalidated__': True}

    stale_getitem = types.FunctionType(
        compile('pass', '<test>', 'exec'),
        invalidated_globals,
        '__getitem__',
    )

    Meta = type('Meta', (type,), {'__getitem__': stale_getitem})
    Float = Meta('Float', (), {})

    _make_module('leaker_mod', {'Float': Float})
    cleanup_modules.append('leaker_mod')

    leakers = adhoc_error._find_leaking_modules()
    assert 'leaker_mod' in leakers

  def test_skips_internal_modules(self, cleanup_modules):
    invalidated_globals = {'__etils_invalidated__': True}

    stale_func = types.FunctionType(
        compile('pass', '<test>', 'exec'),
        invalidated_globals,
        'stale_func',
    )

    _make_module('_private_mod', {'stale_func': stale_func})
    cleanup_modules.append('_private_mod')

    leakers = adhoc_error._find_leaking_modules()
    assert '_private_mod' not in leakers

  def test_no_false_positive_for_clean_module(self, cleanup_modules):
    clean_func = types.FunctionType(
        compile('pass', '<test>', 'exec'),
        {'clean_global': 1},
        'clean_func',
    )

    _make_module('clean_mod', {'clean_func': clean_func})
    cleanup_modules.append('clean_mod')

    leakers = adhoc_error._find_leaking_modules()
    assert 'clean_mod' not in leakers


class TestModuleHasStaleRefs:
  """Tests for _module_has_stale_refs."""

  def test_detects_stale_function(self):
    invalidated_globals = {'__etils_invalidated__': True}

    stale_func = types.FunctionType(
        compile('pass', '<test>', 'exec'),
        invalidated_globals,
        'stale_func',
    )

    module = types.ModuleType('test')
    module.stale_func = stale_func

    assert adhoc_error._module_has_stale_refs(module)

  def test_detects_stale_metaclass(self):
    invalidated_globals = {'__etils_invalidated__': True}

    stale_getitem = types.FunctionType(
        compile('pass', '<test>', 'exec'),
        invalidated_globals,
        '__getitem__',
    )

    Meta = type('Meta', (type,), {'__getitem__': stale_getitem})
    Float = Meta('Float', (), {})

    module = types.ModuleType('test')
    module.Float = Float

    assert adhoc_error._module_has_stale_refs(module)

  def test_ignores_dunder_attrs(self):
    invalidated_globals = {'__etils_invalidated__': True}

    stale_func = types.FunctionType(
        compile('pass', '<test>', 'exec'),
        invalidated_globals,
        '__stale__',
    )

    module = types.ModuleType('test')
    module.__stale__ = stale_func

    assert not adhoc_error._module_has_stale_refs(module)

  def test_no_stale_refs(self):
    module = types.ModuleType('test')
    module.x = 42

    assert not adhoc_error._module_has_stale_refs(module)

  def test_survives_dict_mutation_during_scan(self):
    """Accessing attributes can trigger lazy imports that mutate the dict."""
    module = types.ModuleType('test')

    class LazyTrigger:

      @property
      def __globals__(self):
        module.__dict__['_lazy_resolved'] = True
        return {}

    module.trigger = LazyTrigger()

    # Before the list() snapshot fix, this would raise:
    #   RuntimeError: dictionary changed size during iteration
    result = adhoc_error._module_has_stale_refs(module)
    assert not result
