# Etils

etils (eclectic utils) is an open-source collection of utils for python.

Each top-level submodule is a **self-contained independent** module (with its
own `BUILD` rule), meant to be imported individually. To avoid collisions with
other modules/variables, module names are prefixed by `e` (arbitrary convention):

```python
from etils import epath  # Path utils
from etils import epy  # Python utils
from etils import ejax  # Jax utils
...
```

Becauses each module is independent, only the minimal required libraries are
imported (for example, importing `epy` won't suffer the cost of importing TF,
jax,...)

Documentation:

* `etils.epath`: pathlib-like API for `gs://`, `s3://`,...
* `etils.etree`: Tree utils for `tf.nest`, `jax.tree_utils`, DeepMind `tree`.
* `etils.ecolab`: Colab utils.
* `etils.array_types`: Typing annotations for jax, numpy,... arrays
* `etils.epy`: Collection of generic python utils.

*This is not an official Google product.*
