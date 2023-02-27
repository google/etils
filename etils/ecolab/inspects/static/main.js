/**
 * @fileoverview Tree manager.
 */

/**
 * Load the inner html of a node.
 * @param {string} id_ HTML id of the node
 */
async function load_content(id_) {
  const root = document.getElementById(id_);

  // Guard to only load the content once
  if (!root.classList.contains('loaded')) {
    root.classList.add('loaded');

    // Compute the HTML content in Python
    const html_content = await call_python('get_html_content', [root.id]);

    // Insert at the end, without destroying the one-line content
    root.insertAdjacentHTML('beforeend', html_content);

    // Register listeners for all newly added children
    registerChildrenEvent(root);
  }
}


/**
 * Function which execute the callback.
 * @typedef {function():void} EventCallback
 */

/**
 * Register listerner for all element matching the class name.
 *
 * @param {!HTMLElement} elem Root element to which add the listener
 * @param {string} class_name Class for which add the listener
 * @param {!EventCallback} callback Function to call.
 */
function registerClickListenerOnAll(elem, class_name, callback) {
  const children = elem.querySelectorAll(`.${class_name}`);
  for (const child of children) {
    child.classList.remove(class_name);
    child.addEventListener('click', function(event) {
      event.stopPropagation();  // Only one click processed if multiple listener
      callback.bind(this)();
    });
  }
}

/**
 * Register all listerners for all children.
 * @param {!HTMLElement} elem Root element to which add the listener
 */
function registerChildrenEvent(elem) {
  registerClickListenerOnAll(elem, 'register-onclick-expand', async function() {
    // Do not process the click if text is selected
    const selection = document.getSelection();
    if (selection.type === 'Range') {
      return;
    }

    // TODO(epot): As optimization, it's not required to query the id
    // each time, but instead use closure.
    // TODO(epot): Is there a way to only call this once ?
    await load_content(this.parentElement.id);

    // Toogle the collapsible section
    this.parentElement.querySelector('.collapsible')
        .classList.toggle('collapsible-active');
    this.classList.toggle('caret-down');
  });

  registerClickListenerOnAll(elem, 'register-onclick-switch', function() {
    this.closest('.content-switch').classList.toggle('switch-active');
  });
}