import os
import sys

import tomllib

sys.path.insert(0, os.path.abspath("../src"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Project information
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Pyriodicity"
copyright = "%Y, Iskander Gaba"
author = "Iskander Gaba"
with open("../pyproject.toml", "rb") as f:
    pyproject_data = tomllib.load(f)
    release = pyproject_data["project"]["version"]

# General configuration
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["numpydoc", "autoapi.extension", "sphinx.ext.intersphinx"]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Options for autodoc
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html

autodoc_typehints = "none"

# Options for numpydoc
# https://numpydoc.readthedocs.io/en/latest/install.html#configuration

numpydoc_show_class_members = False

# Options for autoapi
# https://sphinx-autoapi.readthedocs.io/en/latest/reference/config.html

autoapi_dirs = ["../src/pyriodicity"]
autoapi_type = "python"
autoapi_root = "generated"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_keep_files = True
autoapi_add_toctree_entry = False
autoapi_own_page_level = "class"
autoapi_member_order = "groupwise"

# Options for intersphinx
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

# Options for HTML output
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_js_files = ["pypi-icon.js"]
html_sidebars = {"usage": [], "dev": []}
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/iskandergaba/pyriodicity",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/pyriodicity",
            "icon": "fa-custom fa-pypi",
        },
    ],
}
