project = "cpprb"
author = "Hiroyuki Yamada"
copyright = "2018, Hiroyuki Yamada"

extensions = ['sphinx.ext.napoleon']
html_theme = "sphinx_rtd_theme"

html_logo = "../site/static/images/logo.png"
html_favicon = "../site/static/images/favicon.png"

napoleon_include_init_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True

autodoc_default_flags = {
    'members': None,
    'undoc-members': None,
    'special-members': '__init__,__cinit__',
    'show-inheritance': None,
    'member-order': 'bysource',
    'exclude-members': '__dict__,__weakref__,__module__'
}
