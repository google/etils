# Etils

[![Unittests](https://github.com/google/etils/actions/workflows/pytest_and_autopublish.yml/badge.svg)](https://github.com/google/etils/actions/workflows/pytest_and_autopublish.yml)
[![PyPI version](https://badge.fury.io/py/etils.svg)](https://badge.fury.io/py/etils)
[![Documentation Status](https://readthedocs.org/projects/etils/badge/?version=latest)](https://etils.readthedocs.io/en/latest/?badge=latest)

etils (eclectic utils) is an open-source collection of utils for python.

Each top-level submodule is a **self-contained independent** module (with its
own `BUILD` rule), meant to be imported individually. To avoid collisions with
other modules/variables, module names are prefixed by `e` (arbitrary
convention):

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

* [`etils.epath`](https://etils.readthedocs.io/en/latest/epath.html): pathlib-like API for `gs://`, `s3://`,...
* [`etils.etree`](https://etils.readthedocs.io/en/latest/etree.html): Tree utils for `tf.nest`, `jax.tree_utils`, DeepMind `tree`.
* [`etils.enp`](https://etils.readthedocs.io/en/latest/enp.html): Numpy utils.
* [`etils.ecolab`](https://etils.readthedocs.io/en/latest/ecolab.html): Colab utils.
* [`etils.array_types`](https://etils.readthedocs.io/en/latest/array_types.html): Typing annotations for jax, numpy,... arrays
* [`etils.edc`](https://etils.readthedocs.io/en/latest/edc.html): Dataclasses utils.
* [`etils.epy`](https://etils.readthedocs.io/en/latest/epy.html): Collection of generic python utils.
* [`etils.eapp`](https://etils.readthedocs.io/en/latest/eapp.html): Absl flags/app utils.
*  [API design guide](https://etils.readthedocs.io/en/latest/api-design.html).

## Installation

Because each module is independent and require different dependencies, you
can select which modules deps to install:

```sh
pip install etils[array_types,epath,epy]
```

*This is not an official Google product.*
