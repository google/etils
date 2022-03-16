## Colab utils

### Lazy common imports

Running:

```python
from etils.ecolab.lazy_imports import *
```

Will lazilly import in the namespace many common Python packages (jax, tfds,
numpy,...). This has 0 overhead cost as modules are only imported during first
usage.

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

```python
ecolab.lazy_imports.__all__  # List of all modules aliases
ecolab.lazy_imports.LAZY_MODULES  # Mapping <module_alias>: <lazy_module info>
```

Code at:
[lazy_imports.py](https://github.com/google/etils/blob/main/etils/ecolab/lazy_imports.py).


### Display arrays/tensors as images

By running:

```python
ecolab.auto_plot_array()
```

All `(h, w[, c])` jax/numpy/TF arrays bigger than `(10, 10)` will be displayed
as image, without having to manually call `pyplot` .

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
