## Python utils

### Reraise

Reraise an exception with an additional message.

Usage as contextmanager:

```python
with epy.maybe_reraise(f'Error for {x}: '):
  fn(x)
```

Usage using try/except:

```python
try:
  fn(x)
except Exception as e:
  epy.reraise(e, prefix=f'Error for {x}: ')
```

Benefit: Contrary to `raise ... from ...` and `raise
Exception().with_traceback(tb)`, this function will:

*   Keep the original exception type, attributes,...
*   Avoid multi-nested `During handling of the above exception, another
    exception occurred`. Only the single original stacktrace is displayed.

This result in cleaner and more compact error messages.

### Text utils

* `epy.pprint`: Pretty print a value, supports `dataclasses`.
* `epy.pretty_repr`: Returns a pretty `repr()` of the given object, supports
  `dataclasses`.
* `epy.reverse_fstring`: Parse string to match a f-string pattern

  ```python
  epy.reverse_fstring(
      '/home/{user}/projects/{project}',
      '/home/conchylicultor/projects/menhir'
  ) == {
      'user': 'conchylicultor',
      'project': 'menhir',
  }
  ```
* `epy.diff_str`: Pretty diff between 2 objects.
* `epy.dedent`: Like `textwrap.dedent` but also `strip()` the content.

### Itertool utils

* `epy.zip_dict`: Iterate over items of dictionaries grouped by their keys.

  ```python
  d0 = {'a': 1, 'b': 2}
  d1 = {'a': 10, 'b': 20}

  for k, (v0, v1) in epy.zip_dict(d0, d1):
    ...
  ```

* `epy.groupby`: Similar to `itertools.groupby` but return result as a `dict()`

  ```python
  assert epy.groupby(['ddd', 'c', 'aa', 'aa', 'bbb'], key=len) == {
      3: ['ddd', 'bbb'],
      1: ['c'],
      2: ['aa', 'aa'],
  }
  ```

* `epy.splitby`: Split the iterable into 2 lists (false, true), based on the
  predicate.

  ```python
  small, big = epy.splitby([100, 4, 4, 1, 200], lambda x: x > 10)
  assert small == [4, 4, 1]
  assert big == [100, 200]
  ```

### Class utils

* `epy.ContextManager`: Allows to define contextmanager class using
  `yield`-syntax.

  Example:

  ```python
  class A(epy.ContextManager):

    def __contextmanager__(self) -> Iterable[A]:
      yield self


  with A() as a:
    pass
  ```

* `epy.frozen`: Class decorator to make it immutable (except in `__init__`).

  ```python
  @epy.frozen
  class A:

    def __init__(self):
      self.x = 123

  a = A()
  a.x = 456  # AttributeError
  ```

  Supports inheritance, child classes should explicitly be marked as
  `@epy.frozen` if they mutate additional attributes in `__init__`.

* `epy.wraps_cls`: Equivalent of `@functools.wraps`, but applied to classes.
* `epy.is_namedtuple`: Returns `True` if the value is instance of `NamedTuple`.
* `epy.issubclass`: Like `issubclass`, but do not raise error if value is not
  `type`.

### Miscellaneous

* `epy.StrEnum`: Like `enum.StrEnum`, but is case insensitive.
* `epy.lazy_imports`: Context Manager to lazy load modules (for optional
  dependencies or speed-up import time).
* `epy.ExitStack`: Like `contextlib.ExitStack` but supports setting the
  contextmanagers at init time.

### Environment utils

* `epy.is_notebook`: `True` if running inside `Colab` or `IPython` notebook
