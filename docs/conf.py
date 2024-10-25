import os
import sys
import tomllib

sys.path.insert(0, os.path.abspath(".."))

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
    release = pyproject_data["tool"]["poetry"]["version"]

# General configuration
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["numpydoc", "pydata_sphinx_theme", "sphinx.ext.intersphinx"]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Options for autodoc
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html

autodoc_typehints = "none"

# Options for numpydoc
# https://numpydoc.readthedocs.io/en/latest/install.html#configuration

numpydoc_show_class_members = False

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
html_sidebars = {"example": [], "dev": []}
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
            "icon": "fa-brands fa-python",
        },
    ],
}
