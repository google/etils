# Pathlib intro

[TOC]

Have a look at the
[pathlib documentation](https://docs.python.org/3/library/pathlib.html) if
you're not familiar with pathlib.

## Motivation

Pathlib API cleanup many very common file manipulation patterns.

<table>
<tr>
  <th>os.path</th> <th>pathlib</th>
</tr>

<tr>
<td><pre>
path = os.path.join(os.path.dirname(path), 'images')
</pre></td>
<td><pre>
path = path.parent / 'images'
</pre></td>
</tr>
<tr>
<td><pre>
with tf.io.gfile.GFile(path, 'w') as f:
  f.write(content)
</pre></td>
<td><pre>
path.write_text(content)
</pre></td>
</tr>
<tr>
<td><pre>
with tf.io.gfile.GFile(path, 'rb') as f:
  content = f.read()
</pre></td>
<td><pre>
content = path.read_bytes()
</pre></td>
</tr>
<tr>
<td><pre>
if not tf.io.gfile.exists(path):
  tf.io.gfile.mkdirs(path)
</pre></td>
<td><pre>
path.mkdir(parents=True, exist_ok=True)
</pre></td>
</tr>
<tr>
<td><pre>
os.path.basename(path).split('.')[0]
</pre></td>
<td><pre>
path.stem
</pre></td>
</tr>
<tr>
<td><pre>
os.path.splitext(path)[1]
</pre></td>
<td><pre>
path.suffix
</pre></td>
</tr>

</table>

### Convert `Path` -> `str`

Some libraries are not compatible with pathlib. If the library cannot be fixed
to be [PEP 519](https://www.python.org/dev/peps/pep-0519/) compliant, you can
convert pathlib object to `str` with `os.fspath`, like:

```python
path = os.fspath(path)  # Convert `Path` -> `str`
```

### Typing annotations

**Function inputs**: Annotate with `epath.PathLike` to support both `str` and
pathlib objects ( [PEP 519](https://www.python.org/dev/peps/pep-0519/)
compliant):

```python
def save(path: epath.PathLike):
  path = epath.Path(path)  # Normalize `str`,... -> `Path`
  ...


# All the following are valid
save('/some/path')
save('gs://some-bucket/path')
save(pathlib.Path('/some/path'))
save(epath.Path('gs://some-bucket/path'))
```

**Function outputs**: Annotate with `epath.Path`.

## Pathlib mini-intro

To create a path:

```python
path = epath.Path('gs://path/to/my_directory')
```

Most commonly used attributes:

**Attribute** | **Value**
------------- | ---------------------------------
`path`        | `Path('/path/to/file.txt')`
`path.parent` | `Path('/path/to/')`
`path.name`   | `'file.txt'`
`path.suffix` | `'.txt'`
`path.stem`   | `'file'`
`path.parts`  | `('/', 'path', 'to', 'file.txt')`

Most commonly used methods:

*   `path / 'subdir'` (instead of `os.path.join(path, 'subdir')`)
*   `path.exists()`
*   `path.is_dir()`
*   `for p in path.iterdir()`
*   `for p in path.glob('*.jpg')`
*   `path.mkdir()`
*   `path.mkdir(parents=True, exist_ok=True)`

Reading/writing files is also simplified. Instead of `with open():`:

*   `path.write_text(content)`
*   `path.write_bytes(content)`
*   `path.write_text()`
*   `path.read_bytes()`
*   When used with other libraries:

    ```python
    with path.open('rb') as f:
      csv.writer(f)
    ```

To convert:

*   `str` -> pathlib-like: `epath.Path('/my/path')`
*   pathlib-like -> `str`: `os.fspath(my_path)`

See [pathlib doc](https://docs.python.org/3/library/pathlib.html) for more info.
