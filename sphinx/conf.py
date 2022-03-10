project = "cpprb"
author = "Hiroyuki Yamada"
copyright = "2019, Hiroyuki Yamada"

extensions = ['sphinx.ext.napoleon','sphinx_automodapi.automodapi']
html_theme = "sphinx_rtd_theme"

html_logo = "../site/static/images/logo.png"
html_favicon = "../site/static/images/favicon.png"
html_show_sourcelink = False

napoleon_include_init_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True

numpy_show_class_members = False

autodoc_default_options = {
    'member-order': 'bysource',
    'class-doc-from':'both',
    'exclude-members': '__dict__, __weakref__, __module__, __new__, __pyx_vtable__, __reduce__, __setstate__'
}

html_static_path = ['static']
html_css_files = ['custom.css']
