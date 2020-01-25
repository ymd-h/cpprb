project = "cpprb"
author = "Hiroyuki Yamada"
copyright = "2018, Hiroyuki Yamada"

extensions = ['sphinx.ext.napoleon',
              'sphinx_automodapi.automodapi','sphinx_automodapi.smart_resolver']
html_theme = "sphinx_rtd_theme"

html_logo = "../site/static/images/logo.png"
html_favicon = "../site/static/images/favicon.png"

napoleon_include_init_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True

numpy_show_class_members = False

autodoc_default_options = {
    'members': None,
    'member-order': 'bysource',
    'undoc-members': None,
    'private-members': None,
    'inherited-members': None,
    'special-members': '__init__',
    'show-inheritance': None,
    'exclude-members': '__dict__, __weakref__, __module__, __new__, __pyx_vtable__, __reduce__, __setstate__'
}
