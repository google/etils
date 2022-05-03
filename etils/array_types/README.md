## Arrays typing annotations

### API

For aesthetics purpose only, functions input/output can be annotated to help the
reader better understand intended shape/dtype.

```python
from etils.array_types import Array, f32, ui8


def _normalize_image(img: ui8['h w c']) -> f32['h w c']:
  return np.interp(img, from_=(0, 255), to=(-1, 1))
```

This indicates the reader that the function takes a 3d uint8 array and return a
3d float32 with the same shape values.

Note: Those typing annotations are purely aesthetics but are not detected by
static type checking tools. They are only helpful as documentation.

### Annotation conventions

Typing annotations don't have any runtime effect and arbitrary string are
accepted. In practice, for consistency it's best to follow the conventions:

*   Valid symbols:
    *   `str`: Named axis (e.g. `f32['batch height width']`)
    *   `int`: Static axis (e.g. `f32[28, 28]`, `f32['h w 3']`)
    *   `_`: Anonymous axis (e.g. `f32['batch _ _ c']`, `f32[None, 3]`)
    *   `...`: Anonymous zeros or more axis (e.g. `f32['... h w c']`)
    *   `*name`: Named zeros or more axis (e.g. `f32['*batch_dims h w c']`)
    *   `+`, `-`, `/`, `*` operators (e.g. `f32['h/2 w/2 c1+c2']`)
*   Typing annotations are only considered to be consistent per function call,
    so a function `f32['h w'] -> f32['h w']` can be called twice with 2
    different image sizes.
*   Passing multiple values is the same as concatenating the string (e.g.
    `f32[..., 'h', 'w', 3] == f32['... h w 3']`
*   If any dtype is valid, use `Array[...]`.
*   `ArrayLike[f32[...]]` indicates any array convertible values are accepted
    (`list`, `tuple`,...).
