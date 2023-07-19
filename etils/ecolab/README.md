## Colab/Jupyter utils

See the
[demo on Colab](https://colab.research.google.com/github/google/etils/blob/main/etils/ecolab/docs/demo.ipynb).

### Lazy common imports

Running:

```python
from etils.lazy_imports import *
```

Will lazily import in the namespace 100+ common Python packages (jax, tfds,
numpy, functools,...). This has 0 overhead cost as modules are only imported at
first usage.

Some notes:

*   Colab auto-complete & cie will work as expected.
*   Just typing the module name in a cell can trigger an import on the
    background (Colab inspect the names to display metadata, like the link to
    source code & cie).
*   It is recommended to run this before any other import statement so that
    `import *` doesn't overwrite your imports (in case of name collision).
*   If you `adhoc_import` modules already lazy-imported, make sure to call
    `colab_import.reload_package`
*   This should only be used in Colab.

To get the list of available modules:

*   List of all module aliases

    ```python
    lazy_imports.__all__
    ```

*   Mapping `<module_alias>`: `<lazy_module info>`

    ```python
    lazy_imports.LAZY_MODULES
    ```

*   Print the active imports statements (e.g. to convert lazy imports into real
    ones before publishing a notebook)

    ```python
    lazy_imports.print_current_imports()
    ```

Code at:
[lazy_imports.py](https://github.com/google/etils/tree/main/etils/ecolab/lazy_imports.py).

### Display arrays/tensors as images/videos

By running:

```python
ecolab.auto_plot_array()
```

All `([n, ]h, w[, c])` Jax/Numpy/TF/Torch arrays bigger than `(10, 10)` will be
displayed as image(s)/video (if `n > video_min_num_frames` args, default to 15),
without having to manually call `pyplot` .

![https://github.com/google/etils/blob/main/etils/ecolab/docs/imgs/array.png](https://github.com/google/etils/blob/main/etils/ecolab/docs/imgs/array.png?raw=true){height="220"}

The original string representation is still available through `repr(array)`.

### Collapsible logs on colab

Sometimes, you might want to log verbose information (e.g. the content of a
file). To avoid polluting the logs, you can hide the logs inside a collapsible
block (collapsed by default).

```python
with ecolab.collapse('Json content:'):  # Capture both stderr/stdout
  print(json_path.read_text())
```

Example:

![https://github.com/google/etils/blob/main/etils/ecolab/docs/imgs/collapse.png](https://github.com/google/etils/blob/main/etils/ecolab/docs/imgs/collapse.png?raw=true){height="180"}

### Inspect any Python objects

`ecolab.inspect` allow you to interactively explore any Python objects (e.g
`module`, `class`, `dict`,...), with collapsible/expandable sections.

![https://github.com/google/etils/blob/main/etils/ecolab/docs/imgs/inspect.png](https://github.com/google/etils/blob/main/etils/ecolab/docs/imgs/inspect.png?raw=true){height="280"}

When developing interactively on Colab, you can add
`from etils import ecolab ; ecolab.inspect(x)` statements deep inside
your code, executing them will display the visualization on Colab.

To add a button in all cells to transform the last output in:

```python
ecolab.auto_inspect()
```

![https://github.com/google/etils/blob/main/etils/ecolab/docs/imgs/auto_inspect.png](https://github.com/google/etils/blob/main/etils/ecolab/docs/imgs/auto_inspect.png?raw=true){height="70"}

### Inspect nested `dict` / `list`

`ecolab.json` allows you to interactively explore Json nested `dict` / `list`
with collapsible/expandable sections.

![https://github.com/google/etils/blob/main/etils/ecolab/docs/imgs/json.png](https://github.com/google/etils/blob/main/etils/ecolab/docs/imgs/json.png?raw=true){height="180"}

The dict keys and list indices can be filtered from the display field using
regex (e.g. `x.[0-9]` in the above example).

### Syntax highlighting in cell output

Use `ecolab.highlight_html(code_str)` to add Python syntax highlighting to a Python
code string.

Example:

```python
@dataclasses.dataclass
class A:
  x: int

  def _repr_html_(self) -> str:
    from etils import ecolab  # Lazy-import ecolab

    return ecolab.highlight_html(repr(self))

```

![https://github.com/google/etils/blob/main/etils/ecolab/docs/imgs/highlight.png](https://github.com/google/etils/blob/main/etils/ecolab/docs/imgs/highlight.png?raw=true){height="180"}

### Bi-directional Python/Javascript communication

Ecolab provide a simplified API for Python<>Js communication which works for
both `colab` and `jupyter` notebooks.

In Python, use `ecolab.register_js_fn` to register any Python functions. The
function accept any json-like input/outputs (`int`, `str`, `list`, `dict`, `None`,...)

```python
@ecolab.register_js_fn
def my_fn(x, y, z):
  return {'sum': x + y + z}
```

The function can then be called from Javascript with
`call_python('<fn_name>', [arg0, ...], {kwarg0: ..., kwarg1: ...})`

```python
# Currently has to be executed in the same cell to install the library
IPython.display.display(IPython.display.HTML(ecolab.pyjs_import()))

IPython.display.HTML("""
<script>
  async function main() {
    out = await call_python('my_fn', [1, 2], {z: 3});
    console.log(out['sum']);  // my_fn(1, 2, z=3)  == {'sum': 6}
  }
  main();
</script>
""")
```

### Interruptible loops

`ecolab.interruptible` allows graceful interruption of loops. It is especially
useful for slow training loops.

While an iterator wrapped with `interruptible` is running, the first SIGINT
signal (e.g. from Ctrl+C or from interrupting the Colab Kernel) is captured, and
instead of raising an exception the loop simply ends after the current
iteration.

The second SIGINT signal will immediately raise a `KeyboardInterrupt` as usual.

```python
# SIGINT during this loop will finish the current iteration and then
# simply stop without raising an exception raised.
for i in ecolab.interruptible(range(100)):
  time.sleep(2)
  print(i)
```

### Others

*   `ecolab.set_verbose()`: Log stderr & `absl.logging` (which are hidden by
    default)
*   `ecolab.patch_graphviz()`: Make `graphviz` display work on Colab

### Reload modules

Helpful for interactive development to reload from Jupyter notebook the code
we're currently editing (without having to restart the notebook kernel).

Usage:

```python
ecolab.clear_cached_modules(['visu3d', 'other_module.submodule'])

import visu3d
import other_module.submodule
```
