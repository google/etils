## Numpy utils

### Code that works with `np.array`, `jnp.array`, `tf.Tensor`

Use `enp.get_np_module(t)` to write function which works with both `tf`, `jax`
and `numpy`:

```python
def my_function(array: Array):
  xnp = enp.get_np_module(array)
  return xnp.sum(array) + 1


my_function(tf.constant([1, 2]))  # Returns tf.Tensor
my_function(np.array([1, 2]))  # Returns np.ndarray
my_function(jnp.array([1, 2]))  # Returns jnp.ndarray
```

### Interpolation util

`enp.interp` linearly scale an array. API is:
`np.interp(array, from_=(min, max), to=(min, max))`

* Each dimension in the axis can be scaled by a different factor (broadcasting).
* Values outside the boundaries are extrapolated.
* Support `np`, `jnp`, `tnp`

Examples:

* Normalize `np.uint8` image to `np.float32`:

  ```python
  img = enp.interp(img, (0, 255), (-1, 1))
  ```

* Converting normalized 3d coordinates to world coordinates:

  ```python
  coords = enp.interp(coords, from_=(-1, 1), to=(0, (h, w, d)))
  ```

  * `coords[:, 0]` is interpolated from `(-1, 1)` to `(0, h)`
  * `coords[:, 1]` is interpolated from `(-1, 1)` to `(0, w)`
  * `coords[:, 2]` is interpolated from `(-1, 1)` to `(0, d)`
