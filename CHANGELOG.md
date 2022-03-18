# Changelog

## vNext

* `enp`: Expose `enp.lazy` to the public API to help design code compatible
  with `jax`, `np`, `tf`
* `ecolab`:
  * Add `from etils.ecolab.lazy_imports import *` to lazy-import common
  Python modules
  * Rename `ecolab.display_array_as_img` -> `ecolab.auto_plot_array`
  * `ecolab.auto_plot_array` now supports multi-images & video.
* `etree`: Add `assert_same_structure` on the backend
* `epy`: Add `epy.zip_dict`

## v0.4.0

* `edc`: Dataclasses utils
* `etree`:
  * `etree.spec_like`: Inspect a nested structure
* `enp`:
  * `enp.interp`: Scale arrays
  * `enp.normalize_bytes2str`: Normalize `str` arrays
* `ecolab`:
    * Replace `ecolab.collapse_xyz()` by unified `ecolab.collapse()`

## v0.3.3

* Add text utils:
  * `epy.Lines`
  * `epy.dedent`

## v0.3.2

* Automated github release
