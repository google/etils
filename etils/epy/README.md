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

### `epy.frozen`

Class decorator to make it immutable (except in `__init__`).

```python
@epy.frozen
class A:

  def __init__(self):
    self.x = 123

a = A()
a.x = 456  # AttributeError
```

Supports inheritance, child classes should explicitly be marked as `@epy.frozen`
if they mutate additional attributes in `__init__`.
