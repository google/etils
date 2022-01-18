# Etils

[![Unittests](https://github.com/google/etils/actions/workflows/pytest_and_autopublish.yml/badge.svg)](https://github.com/google/etils/actions/workflows/pytest_and_autopublish.yml)
[![PyPI version](https://badge.fury.io/py/etils.svg)](https://badge.fury.io/py/etils)

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

## Documentation

* [`etils.epath`](https://github.com/google/etils/tree/main/etils/epath): pathlib-like API for `gs://`, `s3://`,...
* [`etils.etree`](https://github.com/google/etils/tree/main/etils/etree): Tree utils for `tf.nest`, `jax.tree_utils`, DeepMind `tree`.
* [`etils.enp`](https://github.com/google/etils/tree/main/etils/enp): Numpy utils.
* [`etils.ecolab`](https://github.com/google/etils/tree/main/etils/ecolab): Colab utils.
* [`etils.array_types`](https://github.com/google/etils/tree/main/etils/array_types): Typing annotations for jax, numpy,... arrays
* [`etils.edc`](https://github.com/google/etils/tree/main/etils/edc): Dataclasses utils.
* [`etils.epy`](https://github.com/google/etils/tree/main/etils/epy): Collection of generic python utils.

## Installation

Because each module is independent and require different dependencies, you
can select which modules deps to install:

```sh
pip install etils[array_types,epath,epy]
```

*This is not an official Google product.*
