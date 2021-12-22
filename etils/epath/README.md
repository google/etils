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
