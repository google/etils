## Dataclass utils

`@edc.dataclass(**options)` augment dataclasses with additional
features.

* `replace`: Add a `.replace(` member (alias of `dataclasses.dataclass`)
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
