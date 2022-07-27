## Arrays typing annotations

### API

Function inputs & outputs can be annotated to help the reader better understand
intended shape/dtype.

```python
from etils.array_types import Array, FloatArray, f32, ui8


def _normalize_image(img: ui8['h w c']) -> f32['h w c']:
  return np.interp(img, from_=(0, 255), to=(-1, 1))
```

This indicates the reader that the function takes a 3d uint8 array and return a
3d float32 with the same shape values.

Note: Those typing annotations are not (yet) detected by static type checking
tools. However, they are already helpful as documentation.

### Annotation conventions

Typing annotations shape follow the conventions:

*   Valid symbols:
    *   `str`: Named axis (e.g. `f32['batch height width']`)
    *   `int`: Static axis (e.g. `f32[28, 28]`, `f32['h w 3']`)
    *   `_`: Anonymous axis (e.g. `f32['batch _ _ c']`, `f32[None, 3]`)
    *   `...`: Anonymous zeros or more axis (e.g. `f32['... h w c']`, `f32[...,
        3]`)
    *   `*name`: Named zeros or more axis (e.g. `f32['*batch_dims h w c']`)
    *   `+`, `-`, `/`, `*` operators (e.g. `f32['h/2 w/2 c1+c2']`)
*   Typing annotations are only considered to be consistent **per function
    call**, so a function `f32['h w'] -> f32['h w']` can be called twice with 2
    different image sizes.
*   Passing multiple values is the same as concatenating the string (e.g.
    `f32[..., 'h', 'w', 3] == f32['... h w 3']`
*   DType can be:
    *   `Array[...]`: Any dtype accepted
    *   `FloatArray` (accepts `f32`, `bf16`, ...), `IntArray` (accepts `ui8`,
        `i32`, `i64`, ...): Respectively accept an union of multiple types
    *   `f32`, `ui8`, ...: Specific type
*   `ArrayLike[f32[...]]` indicates any array convertible values are accepted
    (`list`, `tuple`, ...).

### Runtime shape/dtype checking

You can decorate your function with `@enp.check_and_normalize_arrays` so that
array shape/dtype are dynamically validated at runtime:

```python
from etils import enp
from etils.array_types import FloatArray, IntArray


@enp.check_and_normalize_arrays
def add(x: IntArray, y: IntArray) -> IntArray:
  return x + y
```

#### TF / Jax / Numpy compatibility

Functions decorated with `enp.check_and_normalize_arrays` support `np`, `jnp`,
and `tnp`:

*   If args are mixed between `jnp` and `tnp`, an error is raised
*   If args are `xnp` with `np`, the `np` array is auto-casted to `xnp`.
*   You can force usage of TF / Jax / Numpy by passing a `xnp=` kwargs
    (automatically added).

```python
add(np.array(1), jnp.array(2))  # np auto-casted to jnp
add(tf.constant(1), jnp.array(2))  # Error jnp / TF conflict
add(tf.constant(1), jnp.array(2), xnp=jnp)  # Force jnp usage
```

Using `strict=False` makes your function auto-convert `list`, `int`,... to
`xnp.ndarray`:

```python
@enp.check_and_normalize_arrays(strict=False)
def add(x: IntArray, y: IntArray):
  return x + y

add([1, 2, 3], 10)  # == np.array([10, 12, 13])
add([1, 2, 3], 10, xnp=jnp)  # == jnp.array([10, 12, 13])
add([1, 2, 3], tf.constant(10))  # == tnp.array([10, 12, 13])
```

You can add a `xnp: enp.NpModule = ...` kwarg to your function which will be
automatically assigned to the auto-infered xnp:

```python
@enp.check_and_normalize_arrays(strict=False)
def add(x: IntArray, y: IntArray, *, xnp: enp.NpModule = ...):
  return xnp.add(x, y)


add(1, [1, 2, 3])  # Inside the function, `xnp=np`
add(tf.constant(1), tf.constant(2))  # Inside the function, `xnp=tnp`
```

#### DType checking

There are 2 levels of checking:

*   Using type union: `IntArray` (accepts `ui8`, `i32`, `i64`, ...),
    `FloatArray` (accepts `f32`, `bf16`, ...)
*   Using specific type: `f32`, `ui8`, ...

Using type unions allows your functions to support quantization, ...

#### Shape checking

Currently, shape checking is not yet supported (but in project).
