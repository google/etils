## Colab utils

### Lazy common imports

Running:

```python
from etils.ecolab.common import *
```

Will lazilly import in the namespace many common Python packages (jax, tfds,
numpy,...). This has 0 overhead cost as modules are only imported during first
usage.

Colab auto-complete & cie will work as expected. Note that just entering the
module name in a cell might trigger an import on the background (Colab inspect
the names to display metadata, like the link to source code & cie).

The list of imported packages can be seen in:
[common.py](https://github.com/google/etils/blob/main/etils/ecolab/common.py).

Note: This should only be used in Colab.

### Display arrays/tensors as images

By running:

```python
ecolab.display_array_as_img()
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
