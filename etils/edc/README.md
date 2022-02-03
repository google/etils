## Dataclass utils

### Dataclasses validation

`edc.field` is like `dataclasses.field`, but allow to validate and normalize
dataclasses input fields.

```python
@dataclasses.dataclass
class A:
  path: epath.Path = edc.field(validate=epath.Path)
  x: int = edc.field(validate=int)
  y: int = edc.field(validate=lambda x: -x, default=5)


a = A(
   path='/some/path'  # Inputs auto-normalized `str` -> `epath.Path`
   x='123',
)
assert isinstance(a.path, epath.Path)
assert a.x == 123
assert a.y == -5
```

### Augment dataclasses

`@edc.dataclass(**options)` augment dataclasses with additional
features. It works with both `dataclasses.dataclass` and `chex.dataclass`.

* `kw_only`: (False by default) Make the `__init__` only accept keyword
  arguments
* `replace`: Add a `.replace(` member (alias of `dataclasses.dataclass`)
* `repr`: Make the class `__repr__` returns a pretty-printed `str`

  ```python
  @edc.dataclass(repr=True)
  @dataclasses.dataclass
  class A:
    x: Any = None
    y: Any = None

  assert repr(A(123, A(y=456))) == """A(
      x=123,
      y=A(
          x=None,
          y=456,
      ),
  )"""
  ```

* `allow_unfrozen`: Allow to mutate deeply-nested dataclasses by adding `unfrozen()` / `frozen()`
  methods.

  Example:

  ```python
  @edc.dataclass(allow_unfrozen=True)
  @dataclasses.dataclass(frozen=True)
  class A:
    x: Any = None
    y: Any = None

  old_a = A(x=A(x=A()))

  # After a is unfrozen, the updates on nested attributes will be propagated
  # to the top-level parent.
  a = old_a.unfrozen()
  a.x.x.x = 123
  a.x.y = 'abc'
  a = a.frozen()  # `frozen()` recursively call `dataclasses.replace`

  # Only the `unfrozen` object is mutated. Not the original one.
  assert a == A(x=A(x=A(x = 123), y='abc'))
  assert old_a == A(x=A(x=A()))
  ```
