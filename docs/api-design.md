# API design principles

Good practices for designing API, in Python.

## High level

### Start by the end-user code

Before designing any abstraction, start by writing the end-user final code. What
should the final code look like ? (independently of any technical concerns).

It's also helpful to work with a real use-case. Take an existing code, and
rewrite it with your imaginary ideal API to see whether this will actually make
things simpler.

Don't start implementing before having some idea of the end-user API (it does
not need to be final).

### Limit entry points

Ideally, there should be a single way of using the API. Limiting the number of
entry points reduce the mental burden for the user.

Each feature should be kept as a separate entity which can be used
independently.

Example:

*   Originally, TFDS had many ways of loading a dataset:

    ```python
    ds = tfds.load('mnist')
    ds = tfds.load_as_numpy('mnist')

    ds = tfds.builder('mnist').as_dataset()
    ds = tfds.builder('mnist').as_numpy()

    ds = tfds.as_numpy(ds)
    ```

    This added redundancy, unclarity (what's the difference between each
    option?), maintenance burden (to support all options).

    The new API limit the entry points to the minimum:

    ```python
    # Low level API
    ds = tfds.builder('mnist').as_dataset()

    # High level API
    ds = tfds.load('mnist')

    # tf -> np conversion
    ds = tfds.as_numpy(ds)
    ```

*   If 2 arguments are mutually exclusive, it might be a sign they should be
    merged into a single one.

    ```python
    # Rather than:
    fn(shuffle=True, shuffle_seed=123)

    # What is the behavior when `shuffle_seed` is set but `shuffle=False` ?
    # Should it be an error ? Should it silently ignore the seed ?

    # Using a single argument remove the ambiguity
    # By design, users cannot misuse the API
    fn(shuffle=Shuffle(seed=123))
    ```

As new features are added, it is tempting to add new entry points. However, you
should think carefully whether it would be best to instead integrate those with
the existing entry points.

### Don't compromise between simplicity and customizability

Good APIs are simple by default, but powerful when needed.

As more option/features are added, one should be careful about keeping the API
itself simple.

To avoid this, there are multiple strategies.

*   Make sure simple use-cases work out of the box

    Set default arguments / option / behavior to match the most common use-case.
    This means that most users can use the API as-is without any option.

*   Power-users can trade this simplicity for more verbosity.

    *   Complexity should be wrapped behind simpler abstractions.
    *   Use protocols/interfaces to allow users extend your features.

### Inject logic through protocols

Function / class often control multiple features. Rather than implementing the
features directly inside the class, it can be preferable to move the feature
implementation in a separate abstraction.

Example:

```python
# Logic defined inside the implementation
sample(num_points=100, strategy='uniform')

# Logic injected through protocol
sample(num_points=100, strategy=UniformSampling())
```

Externalisation of the implementation allow greater customizability/modularity:

*   User can augment your functions for their custom needs directly in their
    codebase, without having to send a PR (e.g. add a
    `strategy=WeigthedSampling(weights=...)`).
*   Abstractions can be reused in other parts of the code.

Note: It is ok for an argument to accept multiple types. In the above example,
you don't need to choose between accepting `str` or `SamplingProtocol` but can
support both (the `str` for the common use-cases and `SamplingProtocol` for
power-users).

### Features should be self-contained and independent

Rather than monolithic API where all features are entangled together, one should
try to split the API is a collection of self-contained independent abstractions.

Those abstractions can then be composed together (manually by the user, or with
a higher level API). But it should also be possible to use the abstraction
independently (without requiring the higher level API).

Example:

*   Avoid dependencies between abstractions (e.g. between dataset and model).
    This was a main pain of `tensor2tensor` were everything was entangled
    together.
*   `tfds.features` API can be used independently of `tensorflow_datasets`.

Having independent abstractions simplify addoption too, as it allow other users
to reuse only the part they need and integrate them to their existing codebase
(rather than having to migrate their full codebase to the library/frameworks).

Sometimes, those abstractions can even be moved into a new separate independent
module (e.g. `v3d.DataclassArray` -> `dataclass_array.DataclassArray`)

### Use inheritance wisely

The main goal of inheritance should be to define interfaces/protocol. And
eventually to provide a default implementation/behavior.

*   Avoid deep inheritance chain which makes harder to understand which code is
    executed.
*   Avoid overwriting methods implemented in the parent class. Instead, add
    hooks in the parent class to allow either childs to control the behavior, or
    even to remove the need for inheritance entirely.

### Reduce friction

If you notice yourself writing the same boilerplate code snippet over and over,
it likely means it should wrapped in a higher-level abstraction.

Reducing friction was the motivation for many abstractions:

*   Individual colab imports boilerplate replaced by `from
    etils.ecolab.lazy_imports import *`
*   Colab `media.show_images` boilerplate replaced by
    `ecolab.auto_plot_images()`
*   Manual GitHub release automated with `etils-actions/pypi-auto-publish`
    [https://github.com/marketplace/actions/pypi-github-auto-release](https://github.com/marketplace/actions/pypi-github-auto-release)

One limitation is that too many “magic” abstractions can hurt new users to
understand the code. For uncommon operations, it is sometimes best to be more
verbose / explicit.

### API should be self-documenting

Do not rely on people reading the docstring.

Caveats/ambiguities should be made explicit by the API, or raise an explicit
error.

If reading the code does not match user expectations, this should be a bug.

### The first time is never the good one

When designing an API, there will be use-cases that weren't anticipated, or
issues which will conflict with design choices.

API design is an iterative process. The perfect API do not exists. It often
takes multiple trials. Experience from previous projects get accumulated into
new ones:

*   `tensor2tensor` -> `tensorflow_datasets`
*   `jax3d.nerfstatic` ->`visu3d`

### There is no absolute rule

All rules in this doc should be context dependent and have many exceptions.

## Low level

### Use Pythonic patterns

Python has many language features that other languages don't. So good practices
in other languages might not apply to Python.

Writing Python code should use the language features which makes Python more
readable.

*   Avoid unnecessary packaging in favor of simpler imports (use `__init__.py`)
*   Use `@property` (`a.name`) instead of getters/setters (`a.get_name()`)
*   Use contextmanager to factor recurring setup/teardown logic:

    Rather than:

    ```python
    ne = NetworkElement('127.0.0.1')
    try:
      data = ne.fetch_data()
    finally:
      ne.disconnect()
    ```

    Better:

    ```python
    with NetworkElement('127.0.0.1') as ne:
      data = ne.fetch_data()
    ```

*   Use dunder methods:

    *   `__getitem__`: `x.fetch_index(i)` -> `x[i]`
    *   `__len__`: `x.num_elems` -> `len(x)`

### Use `__init__.py`

The public user API should be explicitly defined in your project's `__init__.py`
file.

Having an explicit API has many benefits:

*   Limit entry points: users only have a single import, it makes it obvious how
    to start using the API.
*   Helps discoverability: users can inspect a single file, or rely on
    auto-completion so see which symbols your project provide
*   Force you to be explicit about which features should be used by users and
    which are internal (even a public function might be only meant for internal
    usage only)
*   Having an explicit list of public symbols can make you aware of issue (is
    there too many symbols which would benefit from being factored into a sub
    namespace ?)
*   Helps maintenance: You can refactor internal code while keeping the public
    aliases unchanged

### Prefer kwargs-only argument

Unless the function has some obvious argument order or some main argument,
always use kwargs-only signature.

*   This forces users to write more explicit / readable code
*   This makes it much easier to update the signature (add, reorder, remove
    argument)

Example:

```python
# Rather than:
search('france', 10, True)

# What does the above function do ? The `10` and `True` args are obscure.
# Instead:
search('france', limit=10, reverse=True)
```

The API should **force** kwarg usage using `*`, like:

```python
def search(name: str, *, limit: int, reverse: bool)
```

### Don't return `tuple`

When returning multiple unrelated values, rather than returning tuple, return
structured data, like `dataclass`.

Returning `tuple` is fragile as:

*   User is forced to remember the exact order returned by your function. It is
    painful to use and easy to get wrong
*   If in the future, `fn` signature is updated (e.g. to return additional
    value), this will break all users

Using `dataclass` reduce the mental burden. For example users can rely on
auto-complete to discover the output fields. By design, users cannot get the
argument order wrong.

Example:

```python
# Fragile:
y, loss, grad = fn()

# Instead:
out = fn()  # `fn() -> FnOutput`
out.loss  # User can rely on auto-complete, which help discoverability
out.grad
```

### Reduce verbosity (when possible)

It is a good programing practice to give explicit code (good variable names,
semantic type rather than builtins,...). However, sometimes it might actually
make the API worse by adding verbosity without giving more information.

Example:

*   Names can be shortened to avoid redundancy, or that one part does not bring
    information:
    *   `tree.tree_map` -> `tree.map`
    *   `tfds.features.FeatureConnector` -> `tfds.features.Feature`
*   Function arguments which accept a verbose explicit form can sometimes be
    updated to additionally accept a more user-friendly shortened form. For
    example:

    *   `enum` to also accept their `str` equivalent:

        ```python
        fn(mode=sunds.tasks.YieldMode.STACKED)
        fn(mode='stacked')
        ```

    *   In TFDS, scalar features can be expressed as dtype:

        ```python
        features={'value': tfds.features.Tensor(shape=(), dtype=tf.int64)}
        features={'value': tf.int64}
        ```

    Internally, the short human friendly version gets normalized into the
    boilerplate semantically typed one.

### A bad error message is a bug

Just by looking at the error, users should understand why they got the error and
how to solve it.

*   When the error is raised deeply inside the codebase,
    [`epy.reraise`](https://github.com/google/etils/blob/main/etils/epy/README.md#reraise)
    allow to add high-level context to the error message.
*   Error message should indicates which of function argument is responsible for
    the error
*   When relevant, use hint (Got `y`, did you mean `x`instead ?), using
    `difflib.get_close_matches`
*   This was also the motivation behind `colored_traceback`

### Prefer immutable types

Mutability makes it hard to understand the code without having to deeply inspect
the source code.

Example:

```python
# Rather than:

class MyModel:

  def __call__(self, x):
    self._compute_features(x)
    self._predict()
    return self.y

# Someone who inspects the code will have a hard time understanding what is happening
# It's best to use explicit input/outputs, making it obvious what each function does

class MyModel:

  def __call__(self, x):
    features = self._compute_features(x)
    y = self._predict(features)
    return y

```

### Prefer functions to methods

If a method don't use `self`, it should likely be a function. When someone see
the code:

```python
y = a.fn(x)
```

What does `y` depend of ? Because `fn` is a method, it's unclear whether there
is hidden inputs / outputs (does `fn` has side effects ?). Additionally,
subclass of `A` can overwrite the method, making it harder to understand which
code get actually executed.

By opposition to:

```python
y = fn(x)
```

The function makes it more clear that there's no hidden inputs/outputs.
