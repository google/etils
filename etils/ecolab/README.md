## Colab utils

### Display arrays/tensors as images

By running:

```python
ecolab.display_array_as_img()
```

All `(h, w[, c])` jax/numpy/TF arrays bigger than `(10, 10)` will be displayed
as image, without having to manually call `pyplot` .

![https://i.imgur.com/9XxBUlD.png](https://i.imgur.com/9XxBUlD.png){height="500"}

The original string representation is still available through `repr(array)`.

### Collapsible logs on colab

Sometimes, you might want to log verbose informations (e.g. the content of a
file). To avoid polluting the logs, you can hide the logs inside a collapsible
block (collapsed by default).

```python
with ecolab.collapse_stdall('Json content:'):  # Capture both stderr/stdout
  print(json_path.read_text())
```

Example:

![https://i.imgur.com/KOjUlOg.png](https://i.imgur.com/KOjUlOg.png){height="180"}
