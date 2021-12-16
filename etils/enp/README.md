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
