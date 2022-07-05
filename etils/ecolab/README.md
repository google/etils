## Colab utils

### Lazy common imports

Running:

```python
from etils.ecolab.lazy_imports import *
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

All `([n, ]h, w[, c])` jax/numpy/TF arrays bigger than `(10, 10)` will be
displayed as image(s)/video (if `n > video_min_num_frames` args, default to 15),
without having to manually call `pyplot` .

![https://i.imgur.com/9XxBUlD.png](https://i.imgur.com/9XxBUlD.png){height="500"}

The original string representation is still available through `repr(array)`.

### Collapsible logs on colab

Sometimes, you might want to log verbose informations (e.g. the content of a
file). To avoid polluting the logs, you can hide the logs inside a collapsible
block (collapsed by default).

```python
with ecolab.collapse('Json content:'):  # Capture both stderr/stdout
  print(json_path.read_text())
```

Example:

![https://i.imgur.com/KOjUlOg.png](https://i.imgur.com/KOjUlOg.png){height="180"}
