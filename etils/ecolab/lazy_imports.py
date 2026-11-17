# Copyright 2026 The etils Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common lazy imports.

Usage:

```python
from etils.ecolab.lazy_imports import *
```

To get the list of available modules:

```python
lazy_imports.__all__  # List of modules aliases
lazy_imports.LAZY_MODULES  # Mapping <module_alias>: <lazy_module info>
```
"""

from __future__ import annotations

from etils.ecolab import lazy_utils


def __dir__() -> list[str]:  # pylint: disable=invalid-name
  """`lazy_imports` public API.

  Because `globals()` contains hundreds of symbols, we overwrite `dir(module)`
  to avoid poluting the namespace during auto-completion.

  Returns:
    public symbols
  """
  # If modifying this, also update the `lazy_imports/__init__.py``
  return [
      '__all__',
      'LAZY_MODULES',
      'print_current_imports',
  ]


def print_current_imports() -> None:
  """Display the active lazy imports.

  This can be used before publishing a colab. To convert lazy imports
  into explicit imports.

  For convenience, `from etils.ecolab import lazy_imports` is excluded from
  the current imports.
  """
  print(lazy_utils.current_import_statements(LAZY_MODULES))


_builder = lazy_utils.LazyImportsBuilder(globals())


with _builder.replace_imports(is_std=True):
  # pylint: disable=g-import-not-at-top,unused-import,reimported
  import abc
  import argparse
  import ast
  import asyncio
  import base64
  import builtins
  import collections
  import colorsys
  import copy
  import concurrent.futures
  import contextlib
  import contextvars
  import csv
  import dataclasses
  import datetime
  import difflib
  import dis
  import enum
  import functools
  import gc
  import getpass
  import gzip
  import html
  import inspect
  import io
  import importlib
  import IPython
  import itertools
  import json
  import logging
  import math
  import multiprocessing
  import os
  import pathlib
  import pdb
  import pickle
  import pprint
  import queue
  import random
  import re
  import shutil
  import stat
  import string
  import subprocess
  import sys
  import tarfile
  import textwrap
  import threading
  import time
  import timeit
  import tomllib
  import traceback
  import typing  # Note we do not import `Any`, `TypeVar`,...
  import types
  import urllib
  import uuid
  from unittest import mock
  import warnings
  import weakref
  import zipfile
  # pylint: enable=g-import-not-at-top,unused-import,reimported


with _builder.replace_imports(is_std=False):
  # pylint: disable=g-import-not-at-top,unused-import,reimported
  # ====== Etils ======
  from etils import array_types
  from etils import ecolab
  from etils import edc  # pyrefly: ignore[missing-module-attribute]
  from etils import enp
  from etils import epath
  from etils import epy
  from etils import etqdm
  from etils import etree
  from etils import exm  # pyrefly: ignore[missing-module-attribute]
  from etils import g3_utils
  from etils.ecolab import lazy_imports
  # ====== Common third party ======
  from absl import app
  from absl import flags
  import apache_beam as beam  # pyrefly: ignore[missing-import]
  import bagz  # pyrefly: ignore[missing-import]
  import chex  # pyrefly: ignore[missing-import]
  import dataclass_array as dca  # pyrefly: ignore[missing-import]
  import einops
  import fiddle as fdl  # pyrefly: ignore[missing-import]
  import flask  # pyrefly: ignore[missing-import]
  import flax  # pyrefly: ignore[missing-import]
  from flax import linen as nn  # pyrefly: ignore[missing-import]
  from flax import nnx  # pyrefly: ignore[missing-import]
  import functorch  # pyrefly: ignore[missing-import]
  import gin  # pyrefly: ignore[missing-import]
  import grain.python as grain  # pyrefly: ignore[missing-import]
  import graphviz  # pyrefly: ignore[missing-import]
  import imageio  # pyrefly: ignore[missing-import]
  import immutabledict
  # Even though `import ipywidgets as widgets` is the common alias, widgets
  # is likely too ambiguous.
  import ipywidgets
  import jax  # pyrefly: ignore[missing-import]
  from jax import numpy as jnp  # pyrefly: ignore[missing-import]
  import jaxtyping  # pyrefly: ignore[missing-import]
  import lark  # pyrefly: ignore[missing-import]
  import matplotlib
  import matplotlib as mpl  # Standard alias
  from matplotlib import pyplot as plt
  import mcp  # pyrefly: ignore[missing-import]
  import mediapy as media
  import ml_collections  # pyrefly: ignore[missing-import]
  import networkx as nx  # pyrefly: ignore[missing-source-for-stubs]
  import numpy as np
  import optax  # pyrefly: ignore[missing-import]
  import orbax  # pyrefly: ignore[missing-import]
  from orbax import checkpoint as ocp  # pyrefly: ignore[missing-import]
  from orbax.checkpoint.experimental import v1 as ocp_v1  # pyrefly: ignore[missing-import]
  from orbax import export as oex  # pyrefly: ignore[missing-import]
  import pandas as pd  # pyrefly: ignore[missing-import]
  import PIL
  from PIL import Image  # Common alias
  import pycolmap  # pyrefly: ignore[missing-import]
  import scipy  # pyrefly: ignore[missing-import]
  import seaborn as sns  # pyrefly: ignore[missing-source-for-stubs]
  import sklearn  # pyrefly: ignore[missing-import]
  import tensorflow as tf  # pyrefly: ignore[missing-source-for-stubs]
  import tensorflow.experimental.numpy as tnp  # pyrefly: ignore[missing-import]
  import tensorflow_datasets as tfds  # pyrefly: ignore[missing-import]
  import torch  # pyrefly: ignore[missing-import]
  # from torch import nn  # Collision with flax.linen
  import torchtext  # pyrefly: ignore[missing-import]
  import torchvision  # pyrefly: ignore[missing-import]
  import tqdm
  # tqdm import also trigger additional imports.
  # TODO(epot): Currently pylance might not infer `tqdm.auto` match
  # `import tqdm.auto`
  # Could try to explicitly import inside a `if typing.TYPE_CHECKING:`
  tqdm.auto  # pylint: disable=pointless-statement
  tqdm.notebook  # pylint: disable=pointless-statement
  import tree
  import typeguard  # pyrefly: ignore[missing-import]
  import typing_extensions
  import plotly  # pyrefly: ignore[missing-import]
  from plotly import express as px  # pyrefly: ignore[missing-import]
  from plotly import graph_objects as go  # pyrefly: ignore[missing-import]
  from pydantic import v1 as pydantic
  import requests
  import sunds  # pyrefly: ignore[missing-import]
  import visu3d as v3d  # pyrefly: ignore[missing-import]
  from xmanager.contrib import flow as xmflow  # pyrefly: ignore[missing-import]
  from xmanager import xm
  # pylint: enable=g-import-not-at-top,unused-import,reimported


# Sort the lazy modules per their <module_name>
LAZY_MODULES: dict[str, lazy_utils.LazyModule] = dict(
    sorted(
        _builder.lazy_modules.items(),
        key=lambda x: x[1]._etils_state.module_name,  # pylint: disable=protected-access
    )
)

__all__ = sorted(LAZY_MODULES)  # Sorted per alias
