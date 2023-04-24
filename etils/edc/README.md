## Dataclass utils

### Dataclasses validation

Allow to automatically apply some normalization/validation to the attributes.

```python
@dataclasses.dataclass
class A:
  path: epath.Path = edc.field(validate=epath.Path)
  x: int = edc.field(validate=lambda x: -x, default=5)


a = A(path='/some/path')  # Inputs auto-normalized `str` -> `epath.Path`
assert isinstance(a.path, epath.Path)
assert a.x == -5

a.x = 3
assert a.x == -3  # `validate` applied on all assignments
```

2 forms exists:

*   Using `edc.field`: like `dataclasses.field` but with an additional
    `validate=` kwarg.

    ```python
    @dataclasses.dataclass
    class A:
      path: epath.Path = edc.field(validate=epath.Path)
    ```

    This works on any `@dataclasses.dataclass` (including `frozen=True`).

    `validate=` can be any `Callable[[In], Out]` applied on the input.

*   Using `edc.dataclass` with `edc.AutoCast[T]`. This only works for class
    types.

    ```python
    @edc.dataclass
    @dataclasses.dataclass
    class A:
      path: edc.AutoCast[epath.Path]
    ```

This can be used with `@gin.configurable` to auto-convert builtins (`str`,
`int`, ...) from the config into their corresponding semantic type
(`pathlib.Path`, `MyEnum`,...) in Python.

### Mutate nested frozen dataclasses

`allow_unfrozen`: Allow to mutate deeply-nested dataclasses by adding
`unfrozen()` / `frozen()` methods.

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

### Wrap fields around ContextVar

Fields annotated with `edc.ContextVar` will be wrapped around
`contextvars.ContextVar`.

Afterward each thread / asyncio coroutine will have its own version of the
fields (similarly to `threading.local`).

The contextvars are lazily initialized at first usage. They do not need to
have default values.

Example:

```python
@edc.dataclass
@dataclasses.dataclass
class Context:
  thread_id: edc.ContextVar[int] = dataclasses.field(
      default_factory=threading.get_native_id
  )

  # Local stack: each thread will use its own instance of the stack
  stack: edc.ContextVar[list[str]] = dataclasses.field(default_factory=list)

  # Shared stack: Instance shared across all threads / coroutines
  shared_stack: list[str] = dataclasses.field(default_factory=list)

# Global context object
context = Context(thread_id=0)

def worker():
  # Inside each thread, the worker use its own context
  assert context.thread_id != 0
  context.stack.append(1)
  time.sleep(1)
  assert len(context.stack) == 1  # Other workers do not modify the local stack

with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
  for _ in range(10):
    executor.submit(worker)
```

### Other augmentations

`@edc.dataclass(**options)` augment dataclasses with additional features. It
works with both `dataclasses.dataclass` and `chex.dataclass`.

*   `kw_only`: (False by default) Make the `__init__` only accept keyword
    arguments
*   `replace`: Add a `.replace(` member (alias of `dataclasses.dataclass`)
*   `repr`: Make the class `__repr__` returns a pretty-printed `str`

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
