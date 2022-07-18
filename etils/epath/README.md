## Pathlib-like API

### Full API

*   `Path(*parts: PathLike)`: pathlib-like abstraction.

Resource API:

*   `resource_path(module_name: str | ModuleType) -> Path`: Resource path
    (read-only).
*   `to_write_path(path: Path) -> Path`: Convert read-only resource path into
    writable path.

Typing API:

*   `PathLike`: `str` or pathlib object (typing annotation): `path: PathLike`
*   `PathLikeCls`: `str` or pathlib object (at runtime): `isinstance(p,
    PathLikeCls)`

### Additional methods

In addition of the pathlib methods, `epath.Path` has the additional methods:

*   `path.copy(dst, overwrite=True)`
*   `path.rmtree()`
*   `path.format(*args, **kwargs)`: Apply `str.format` on the underlying path
    `str`

There are some [discussions](https://github.com/python/cpython/issues/92771)
about adding those methods nativelly in pathlib.

### FLAGS

`absl.flags` support is provided to define `FLAGS` parsed as `epath.Path`:

```python
from absl import app
from etils import epath

_PATH = epath.DEFINE_path('path', None, 'Path to input file.')


def main(_):
  content = _PATH.value.read_text()

if __name__ == "__main__":
  app.run(main)
```
