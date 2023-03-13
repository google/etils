/**
 * @fileoverview Bi-directional Python communications.
 */

// TODO(epot): Also support non-Colab jupyter notebook

/**
 * Call a Python function from Javascript.
 *
 * @param {string} fn_name Function name. Should be registered with
 * `register_js_fn`
 * @param {!Array=} args Args passed to `fn_name(*args)`
 * @param {!Object=} kwargs Kwargs passed to `fn_name(**kwargs)`
 * @returns {!Object} Output of the function `out = fn_name()` (json-like
 * structure)
 */
async function call_python(fn_name, args = [], kwargs = {}) {
  const out = await google.colab.kernel.invokeFunction(fn_name, args, kwargs);
  const data = out.data['application/json'];
  if ('__etils_pyjs__' in data) {
    return data['__etils_pyjs__'];  // Unwrap inner value
  } else {
    return data;
  }
}
