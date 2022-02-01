## Tree utils

`etree` can be used with 3 different backends, depending on usage:

```python
from etils.etree import jax as etree  # Backend: jax.tree_utils
from etils.etree import nest as etree  # Backend: tf.nest
from etils.etree import tree as etree  # Backend: tree (DeepMind)
```

### parallel_map

Similar to `tree.map_structure`, but each leaf is executed in parallel.

```python
img_paths = {'train': ['img0.png', ...], 'test': ['img1.png', ...]}
imgs = etree.parallel_map(imageio.imread, img_paths)  # Load images in parallel
```

Kwargs:

*  `progress_bar`: If `True`, display a progress bar
*  `num_threads`: Number of parallel threads (default to number of CPUs * 5)

### unzip

Unpack a tree of iterable. This is the reverse operation of `tree.map_structure(zip, *trees)`

Example:

```python
etree.unzip({'a': np.array([1, 2, 3])}) == [{'a': 1}, {'a': 2}, {'a': 3}]
```

### spec_like

Extract the shape/dtype of a tree of array. Used for inspecting:

```python
model = MyModel()
variables = model.init(jax.random.PRNGKey(0), x)

# Inspect the `variables` tree structures
print(etree.spec_like(variables))
```

Output the shape/dtype of all elements:

```python
FrozenDict({
    params: {
        Conv_0: {
            kernel: f32[1 1 3 32],
            bias: f32[32],
        },
        Conv_1: {
            kernel: f32[1 1 32 32],
            bias: f32[32],
        },
[...]
```
This works with np, jax, tf arrays.