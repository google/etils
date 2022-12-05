# Changelog

<!--

Changelog follow https://keepachangelog.com/ format.

-->

## [Unreleased]

*   `etree`:
    *   Added: `etree.stack` to stack/batch multiple trees of arrays together.
*   `enp`:
    *   Added: `enp.batch_dot`: Always dot product on the last axis with
        broadcasting support (while `np.dot` is inconsistent 1-D vs 2-D).
    *   Added: `enp.angle_between` to compute angle between 2 n-dimensions
        vectors.
    *   Changed: `enp.project_onto_vector`, `enp.project_onto_plane` supports
        broadcasting.
*   `ecolab`:
    *   Added: `ecolab.patch_graphviz` to fix Colab display with `graphviz`.
    *   Added: `ecolab.set_verbose` to activate stderr logging on Colab.
    *   Changed: `ecolab.clear_cached_modules` accept single `str`
    *   Changed: `ecolab.clear_cached_modules` has a `invalidate=False` to not
        invalidate previous instances of the modules.
*   `edc`:
    *   Added: Expose `edc.repr`, for functional use or directly assign class
        members (e.g. `__repr__ = edc.repr`)
*   `eapp`:
    *   Fixed: `eapp.better_logging()` do not raise `is_borg` `AttributeError`
        anymore.
*   `epath`:
    *   Fixed: `local_path.copy('gs://')` uses the correct backend.
*   All:
    *   Changed: Better error message when missing import.

<!-- --------- Do not add anything bellow this line --------- -->

## [0.9.0] - 2022-10-28

*   `eapp` (Added):
    *   Added: `.make_flags_parser` to define CLI flags through dataclasses.
    *   Added: `.better_logging` to display logs by default, tqdm
        compatibility,...
*   `epy`:
    *   Added: `@epy.frozen` class decorator to make class immutable
*   `edc`:
    *   Added: Attributes of `@edc.dataclass` can be annotated with `x:
        edc.AutoCast[T]` (for auto-normalization).
*   `ecolab`:
    *   Added: `ecolab.clear_cached_modules` to reload modules (useful for
        interactive development)
    *   Added: `etils.lazy_imports` supports multi-import (e.g. using
        `concurrent` also trigger `concurrent.futures` import).

## [0.8.0] - 2022-09-12

*   `epath`:
    *   Added: `path.stat()`
    *   Fix performance issues for `path.mkdir`.
*   `ecolab`:
    *   Added: `from etils.lazy_imports import *` alias of `from
        etils.ecolab.lazy_imports import *`.
    *   Changed: Mutating a lazy_import module mutate the original one. This
        allow to mutate `builtins` module for example.
*   `edc`:
    *   Changed: `.replace` only added once (not in subclasses).

## [0.7.1] - 2022-08-09

*   `ecolab`: Added `dataclass_array` to lazy_imports.

## [0.7.0] - 2022-08-08

*   `array_types`:
    *   Added: More array types: `complex64`,...
    *   Added: An **experimental** `array_types.dtypes.DType` to support more
        flexible dtype expression (`AnyFloat`, type union,...)
    *   Changed: `FloatArray`, `IntArray` do supports any float, int without
        casting.
    *   Changed (breaking): Array dtypes (e.g. (`f32.dtype`) are now
        `array_types.dtypes.DType`.
*   `ecolab`:
    *   Added: `lazy_imports.print_current_imports` to display the active lazy
        imports (e.g. to add imports before publishing a colab).
*   `epy`:
    *   Added: `epy.ContextManager` to create yield-based contextmanager class
        (see
        [discussion](https://discuss.python.org/t/yield-based-contextmanager-for-classes/8453))
    *   Added: `epy.issubclass` (like `issubclass` but does not raises error for
        non-types)
    *   Added: `epy.groupby`, like `itertools.groupby` but returns a `dict`.
    *   Added: `epy.Lines.make_block` helper to create code blocks (function
        calls,...)
    *   Fixed: `epy.StrEnum` raises better error message if invalid input.
*   `epath`
    *   Added: `epath.DEFINE_path` for `absl.flags` support
    *   Changed (breaking): Recursive glob (`rglob`, `glob('**/')`) now raise an
        error rather than being silently ignored.
    *   Changed: `path.as_uri()` returns `gs://` and `s3://` (rather than
        `file:///gs/`)
    *   Changed: Add `__eq__` and `__hash__` for resource path.
*   `edc`
    *   Fixed: `__repr__` is correctly added in Python 3.10 (#143)
    *   Fixed: `dc.frozen()` compatibility with autograd.
    *   Changed: `dc.unfrozen()`now supports `jax.tree_map`.
    *   Changed: Better `dc.unfrozen()` repr which display overwritten fields.
*   `enp`:
    *   Added: `enp.check_and_normalize_arrays` util to dynamically validate
        array dtype/shape from typing annotations.
    *   Added: `enp.linalg.normalize` util.
    *   Added: `enp.project_onto_vector` and `enp.project_onto_plane` geometry
        utils.
*   Other:
    *   Added: Guide on
        [API design principles](https://github.com/google/etils/blob/main/docs/api-design.md).

## [0.6.0] - 2022-05-31

*   `epath`:
    *   Remove TensorFlow dependency from `epath.Path` by default. (For now
        accessing `gs://` still require TF to be installed).
    *   Add `epath.testing.mock_epath` to mock GCS calls.
*   `epy.testing`:
    *   Add `epy.testing.subtest` for better subtests support.
    *   Add `epy.testing.non_hermetic` to mark non-hermetic tests.
*   `oss-kit`:
    *   [`pypi-auto-publish`](https://github.com/marketplace/actions/pypi-github-auto-release)
        GitHub action for automated PyPI and GitHub releases

## [0.5.1] - 2022-05-04

*   `array_utils`:
    *   Now has a `__all__` to allow `from etils.array_types import *`
    *   `FloatArray` and `IntArray` have dtype to `np.float32` and `np.int32`
*   `ecolab`:
    *   Add `with ecolab.adhoc():`contextmanager for dynamic code import.
    *   More `ecolab.lazy_imports`
*   `enp`:
    *   Add `enp.linalg` alias of `enp.compat`
    *   `enp.testing.parametrize_xnp` now has a `restrict` kwarg
*   `epy`:
    *   `StrEnum` is now case-insensitive

## [0.5.0] - 2022-03-23

*   `enp`:
    *   Expose `enp.lazy` to the public API to help design code compatible with
        `jax`, `np`, `tf`
    *   Add `enp.compat` for ops compatibles with both TF, Jax, Np
    *   Add `enp.tau` [constant](https://tauday.com/)
    *   Add `enp.testing.parametrize_xnp` fixture to test a function with both
        `jax`, `tf`, `np`
    *   Add `enp.testing.set_tnp` fixture to activate tf numpy mode.
*   `ecolab`:
    *   Add `from etils.ecolab.lazy_imports import *` to lazy-import common
        Python modules
    *   Rename `ecolab.display_array_as_img` -> `ecolab.auto_plot_array`
    *   `ecolab.auto_plot_array` now supports multi-images & video.
*   `etree`: Add `assert_same_structure` on the backend
*   `epy`: Add `epy.zip_dict`

## [0.4.0] - 2022-02-03

*   `edc`: Dataclasses utils
*   `etree`:
    *   `etree.spec_like`: Inspect a nested structure
*   `enp`:
    *   `enp.interp`: Scale arrays
    *   `enp.normalize_bytes2str`: Normalize `str` arrays
*   `ecolab`:
    *   Replace `ecolab.collapse_xyz()` by unified `ecolab.collapse()`

## [0.3.3] - 2022-01-07

*   Add text utils:
    *   `epy.Lines`
    *   `epy.dedent`

## [0.3.2] - 2022-01-04

*   Automated github release

[Unreleased]: https://github.com/google/etils/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/google/etils/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/google/etils/compare/v0.7.0...v0.8.0
[0.7.1]: https://github.com/google/etils/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/google/etils/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/google/etils/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/google/etils/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/google/etils/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/google/etils/compare/v0.3.3...0.4.0
[0.3.3]: https://github.com/google/etils/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/google/etils/releases/tag/v0.3.2
