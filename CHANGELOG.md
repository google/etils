# Changelog

<!--

Changelog follow https://keepachangelog.com/ format.

-->

## [Unreleased]

*   `epy`:
    *   `epy.lazy_api_imports`: Make lazy importing thread safe.

## [1.11.0] - 2024-11-27

*   `enp`:
    *   Make `enp.testing.parametrize_xnp()` import only requested xnp modules.
    *   Fix orbax error when inspecting specs of an orbax checkpoint.
*   `ecolab`:
    *   `ecolab.inspect`: Proto are better displayed (hide attributes
        `DESCRIPTOR`, `Extensions` in sub-section)
*   `epy`:
    *   `epy.lazy_api_imports`: Fix infinite recursion when importing sub-module
*   `exm`:
    *   Add dummy implementation of the API to simplify open-sourcing.

## [1.10.0] - 2024-10-17

*  `epy`:
    *   Add frozen dataclass support for `epy.ContextManager`
    *   Make `epy.StrEnum` truly case-insensitive
    *   Support adhoc import of proto files with hyphen.
    *   Add `fiddle` supports to `epy.pretty_repr`
*  `enp`: Add `ArraySpec` support for `grain.python.SharedMemoryArrays`.

## [1.9.4] - 2024-09-03  <!-- Should have been 1.10.1 -->

*   Return Python 3.10 support.

## [1.9.3] - 2024-08-30  <!-- Should have been 1.10.0 -->

*   `eapp`:
    *   Allow extra kwargs in `eapp.make_flags_parser()`
*   `epath`:
    *   Fix epath.Path pydantic deserialization for URI-style paths
*   `epy`:
    *   Add `epy.is_test` to check whether we're running in a test environment.
    *   Add `epy.typing.Json`.
    *   Add `epy.ExitStack` which allows setting the contextmanagers during init.
    *   Add proto support for `epy.binary_adhoc`
    *   Hide reraise from the traceback for cleaner error messages
*   `exm`:
    *   Add `exm.url_to_python_only_logs()` and `exm.curr_job_name()` to add
        artifact to Python only logs (without all the verbose C++ junk ).
    *   Fix a bug which makes `exm.current_experiment` crash

## [1.9.2] - 2024-06-12

*   `epath`:
    *   Support pydantic serialization of epath.Path

## [1.9.1] - 2024-06-04

*   `epath`:
    *   Fix an infinite recursion on `is_relative_to` for Python>=3.12.

## [1.9.0] - 2024-05-31

*   `epy`:
    *   Add `epy.lazy_api_imports` to lazy-import `__init__.py` symbols.
    *   Removed: `epy.cached_property`
    *   `epy.lazy_imports`: Error callback accept a `str` to auto-re-raise
        with additional info.
    *   Fix mixing `epy.lazy_imports()` with `epy.binary_adhoc()`.
*   `ecolab`:
    *   Added `reload_workspace=True` to adhoc to auto-reload from workspace
    *   Add `ecolab.get_permalink()`
    *   Fix `ecolab.inspect` not finding static files when the kernel contain
        partial etils deps.
*   `epath`:
    *   Fix error when `importlib.resources.files` return `MultiplexedPath`
    *   Fix `gs://` URI for 3.12
    *   Fix `.walk` 3.12 error (`topdown` -> `top_down` rename)
*   Full compatibility with Python 3.12 (unit tests run on both 3.11 and 3.12).

## [1.8.0] - 2024-03-13

*   Drop Python 3.10 support.
*   `epy`:
    *   `epy.pretty_repr`: Add support for namedtuple
*   `ecolab`:
    *   Add `ecolab.disp(obj)`
    *   Add `;h` for syntax highlighting with auto-display
    *   Fix proto error on import

## [1.7.0] - 2024-02-15

*   `epath`:
    *   Add `mode` to `epath.Path.stat` output. Does not work for Windows nor
        when `tf.io.gfile` is used.
    *   Add `.walk` to `epath.Path`. Similar usage than `pathlib.Path.walk`
*   `epy`:
    *   Added: `epy.reverse_fstring`: Reverse fstring parsing
    *   Added: `reload=` for `epy.binary_adhoc()`, fixed behavior for
        consistency with `ecolab.adhoc`
    *   Added: `epy.pprint`: Pretty print an object (including dataclass).
    *   Added: `epy.pretty_repr_top_level`
    *   Added: `epy.wraps_cls` equivalent of `functools.wraps` but for classes.
    *   Breaking: `epy.lazy_imports(error_callback=)` has now signature
        `(Exception) -> None` (instead of `(str) -> None`)
    *   Fixed: `epy.pretty_repr` missing trailing `,` for tuple with single
        element.
*   `ecolab`:
    *   Changed: `ecolab.auto_display`: Better representation when line is
        displayed
    *   Fix `adhoc` that delete sub-module when `invalidate=False`
    *   `adhoc` with `reload_mode=UPDATE_INPLACE` now supports enums, so old
        versions compare equal to new versions. Enums compared as `a is FOO`
        might still fail.
    *   `adhoc` with `reload_mode=UPDATE_INPLACE` is now much faster.
    *   When using `cell_autoreload=True` the default `reload_mode` is
        now `UPDATE_INPLACE`.
    *   Better error message for adhoc reload
*   `exm`:
    *   Added: `exm.set_citc_source()` to specify which workspace to use when
        using XManager on Colab
*   `etree`:
    *   Added `is_leaf` kwarg to `.map` and `.parallel_map`
*   `enp`:
    *   Add `ArraySpec` support for `flax.linen.summary`.

## [1.6.0] - 2023-12-11

*   `ecolab`:
    *   Added protobuf repeated fields support to `ecolab.inspect`
    *   `ecolab.auto_display`:
        *   Support specifiers to customize auto-display (`;s`, `;a`, `;i`,
            `;p`,...)
        *   Fixed auto-display when the line contain UTF-8 character
    *   Fix a bug for `ecolab.highlight_html` to escape HTML string.
*   `epath`:
    *   `path.mkdir` now supports `mode=` (for `os.path` and `gfile` backend).
*   `epy`:
    *   `epy.lazy_imports()` support adhoc imports (will re-create the original
        `ecolab.adhoc` context when resolved)
    *   Added: `epy.binary_adhoc()` to add adhoc imports when using Python
        interpreter.
    *   `epy.lazy_imports()` supports error and success callbacks
        (`error_callback`/`success_callback`).
    *   Added: `epy.pretty_repr` support `attr` dataclass-like objects.
*   `exm`:
    *   Adding artifacts can be used inside `with xm.create_experiment()`

## [1.5.2] - 2023-10-24

*   `ecolab`:
    *   Fix import error in IPython when 7.0 &lt;= version &lt; 8
*   `epath`:
    *   Fix resource_path when used on a adhoc-imported module.

## [1.5.1] - 2023-10-10

*   `epath`:
    *   Fix `glob` issue when used with ffspec.

## [1.5.0] - 2023-09-19

*   `ecolab`:
    *   Auto display statements ending with `;` (assignments, return
        statements, expressions,...).
    *   Adhoc proto now supports message extensions.
*   `epath`:
    *   Add `missing_ok=False` kwargs to `path.rmtree`.
    *   Add `fsspec_backend` relying on fsspec to handle GCS/S3 without needing
        TensorFlow. This means that `gcsfs` and `s3fs` become required
        dependencies to read respectively GCS and S3. TensorFlow is no more
        required. **Note**: If TensorFlow is installed, we still default to
        the `tf_backend` for backward compatibility.
    *   Changed: `path.glob` raise an error if insufficient permission.
*   `enp`:
    *   `ArraySpec.from_array`:
        *   Fix when TF is in graph mode.
        *   Add `orbax` metadata support
*   `epy`:
    *   Add `epy.pretty_repr` for pretty print an object (including dataclass).
    *   Add `epy.diff_str` for pretty diff print of 2 objects.
    *   Add `epy.is_namedtuple`
*   `etree`:
    *   Fix `etree.map` for `namedtuple` (Python backend)

## [1.4.1] - 2023-07-31

*   `lazy_imports`: More lazy imports
*   `epath`: Remove circular deps

## [1.4.0] - 2023-07-25

*   `epy`:
    *   Add `@epy.lazy_imports` to lazyly import modules.
    *   Fix `@epy.frozen` when class has custom `__getattr__`
*   `ecolab`:
    *   `ecolab.collapse()`
        *   Breaking: Remove `widget=True` argument to  (widget always enabled).
            *   Add `expanded: bool` kwargs to control whether the widget is
                expanded or collapsed by default.
    *   Breaking: Remove `keep_proto` kwargs from `clear_cached_modules` and
        `import_proto` kwargs from `adhoc` (proto always supported)
    * Add `expanded` argument to `ecolab.json()` and change default behavior
      to False.
*   `enp`:
    *   Make array spec (e.g. `etree.spec_like()`) hashable.
    *   `enp.lazy.is_tf` returns `True` for `tf.TensorSpec`
    *   Remove `array_types.dtypes` and `array_types.typing` (should use
        `enp.dtypes` and `enp.typing` instead).
*   `epath`:
    *   Add `owner` and `group` to `epath.Path.stat` output. Does not work for
        Windows nor when `tf.io.gfile` is used.
*   `etree`:
    *   Fix `etree.map` for `collections.defaultdict`
*   `internal`:
    *   Add a `unwrap_on_reload` to save/restore original function after a
        module is reloaded (e.g. on colab)

## [1.3.0] - 2023-05-12

*   `ecolab`:
    *   Add `widget=True` argument to `ecolab.collapse()` for better
        interactivity.
    *   Add `ecolab.highlight_html(code)` to add syntax highlighting to cell
        outputs of specific objects.
*   `edc`:
    *   Add `contextvars` option: Fields annotated as `edc.ContextVars[T]` are
        wrapped in `contextvars.ContextVars`.
    *   Fix error when using `_: dataclasses.KW_ONLY`
    *   `__repr__` now also pretty-print nested dataclasses, list, dict,...
*   `enp`:
    *   `ArraySpec` support `grain.ArraySpec` (allow support
        `etree.spec_like(ds.element_spec)` on grain datasets)
*   `epy`:
    *   Better `epy.Lines.make_block` for custom pretty print classes, list,...

## [1.2.0] - 2023-04-03

*   `enp`:
    *   `etree.spec_like` support `f32`,... annotations (when fully defined).
*   `etree`:
    *   Add `optree` backend.
*   `ecolab`:
    *   Add `ecolab.auto_inspect()` to allow inspecting all colab outputs.
    *   Fix various `ecolab.inspect` issues (e.g. when used on metaclasses,...).
        *   Add `ecolab.inspect` support for Jupyter notebook (non-Colab).

## [1.1.1] - 2023-03-20

*   `enp`:
    *   Fix `torch==2.0` compatibility.
*   Fix warning during `pip install` (missing `epath-no-tf`)

## [1.1.0] - 2023-03-13

*   `enp`:
    *   **Add `torch` support!**
    *   Add `enp.lazy.LazyArray` to lazily check `isinstance(array,
        enp.lazy.LazyArray)` without triggering `TF`, `jax`,... imports
    *   Add `skip=['jnp', ...]` kwarg to `enp.testing.parametrize_xnp` to
        exclude a specific xnp module from tests.
    *   Add `enp.compat` function to fix compatibility issues between Jax, TF,
        torch, numpy (e.g. `x.astype()` missing from `torch`, but working in
        `jax`, `tf`, `np`).
    *   Breaking: Move some functions from `enp.linalg` to `enp.compat`.
    *   Add more `enp.lazy.` methods for conversions from/to dtype
*   `ecolab`:
    *   Add `ecolab.inspect` for interactively inspect any Python objects.
    *   Add `ecolab.json` for interactive expandable JSON display.
    *   `ecolab.auto_plot_array`
        *   Add `torch`
        *   Small/large images (height outside 100-250px) are automatically
            scaled up/down. Can be disabled with
            `ecolab.auto_plot_array(height=None)`
        *   Can overwrite `mediapy` default options with `show_images_kwargs`
            and `show_videos_kwargs`
    *   Add `ecolab.interruptible` for graceful interruption of loops.
    *   Fixed: Pytype/pylance support for `lazy_imports`: unlock auto-complete,
        docstring tooltip, do not trigger linter errors anymore (`"xxx" is not
        definedPylancereportUndefinedVariable`).
*   `etree`:
    *   `from etils import etree` now expose the Python backend (
        `etree.map`,...). Other backend are still available as previously
        (`etree.jax.map`,...)
*   `epy`:
    *   Added: `epy.splitby` to split an iterator in 2, based on a predicate.
        (e.g. `small, big = epy.splitby([20, 1, 30, 1, 2], lambda x: x > 10)`)

## [1.0.0] - 2023-01-09

*   `etree`:
    *   Added: `etree.stack` to stack/batch multiple trees of arrays together.
    *   Added: `etils.etree.py` backend (pure Python implementation to avoid
        installing extra deps).
    *   Added: `etree.map` (as convenience to access the corresponding
        `map_structure`).
*   `enp`:
    *   Added: `enp.batch_dot`: Always dot product on the last axis with
        broadcasting support (while `np.dot` is inconsistent 1-D vs 2-D).
    *   Added: `enp.angle_between` to compute angle between 2 n-dimensions
        vectors.
    *   Changed: `enp.project_onto_vector`, `enp.project_onto_plane` supports
        broadcasting.
    *   Fixed: `TF` accidentally imported when using `enp.linalg` functions.
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
                    *   Added: `epy.issubclass` (like `issubclass` but does not
                        raises error for non-types)
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
    *   Changed: `dc.unfrozen()`now supports `jax.tree.map`.
    *   Changed: Better `dc.unfrozen()` repr which display overwritten fields.
*   `enp`:
    *   Added: `enp.check_and_normalize_arrays` util to dynamically validate
        array dtype/shape from typing annotations.
    *   Added: `enp.linalg.normalize` util.
    *   Added: `enp.project_onto_vector` and `enp.project_onto_plane` geometry
        utils.
*   Other:
    *   Added: Guide on [API design
        principles](https://github.com/google/etils/blob/main/docs/api-design.md).

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

[Unreleased]: https://github.com/google/etils/compare/v1.11.0...HEAD
[1.11.0]: https://github.com/google/etils/compare/v1.10.0...v1.11.0
[1.10.0]: https://github.com/google/etils/compare/v1.9.4...v1.10.0
[1.9.4]: https://github.com/google/etils/compare/v1.9.3...v1.9.4
[1.9.3]: https://github.com/google/etils/compare/v1.9.2...v1.9.3
[1.9.2]: https://github.com/google/etils/compare/v1.9.1...v1.9.2
[1.9.1]: https://github.com/google/etils/compare/v1.9.0...v1.9.1
[1.9.0]: https://github.com/google/etils/compare/v1.8.0...v1.9.0
[1.8.0]: https://github.com/google/etils/compare/v1.7.0...v1.8.0
[1.7.0]: https://github.com/google/etils/compare/v1.6.0...v1.7.0
[1.6.0]: https://github.com/google/etils/compare/v1.5.2...v1.6.0
[1.5.2]: https://github.com/google/etils/compare/v1.5.1...v1.5.2
[1.5.1]: https://github.com/google/etils/compare/v1.5.0...v1.5.1
[1.5.0]: https://github.com/google/etils/compare/v1.4.1...v1.5.0
[1.4.1]: https://github.com/google/etils/compare/v1.4.0...v1.4.1
[1.4.0]: https://github.com/google/etils/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/google/etils/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/google/etils/compare/v1.1.1...v1.2.0
[1.1.1]: https://github.com/google/etils/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/google/etils/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/google/etils/compare/v0.9.0...v1.0.0
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
