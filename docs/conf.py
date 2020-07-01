# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
import os
import sys
# from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath('.'))

#######
sys.path.insert(0, os.path.abspath('../'))

# -- Project information -----------------------------------------------------

project = 'Test nbsphinx'
copyright = '2020, me'
author = 'me'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
   # 'sphinx.ext.napoleon',
    'nbsphinx',
    'sphinx.ext.mathjax',
  #  'sphinx_gallery.load_style',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

highlight_language = 'none'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

nbsphinx_allow_errors = True

nbsphinx_epilog = """
----

Generated by nbsphinx_ from a Jupyter_ notebook.

.. _nbsphinx: https://nbsphinx.readthedocs.io/
.. _Jupyter: https://jupyter.org/
"""

# nbsphinx_thumbnails = {
#     'gallery/*-rst': '_static/copy-button.svg',
# }

# latex_elements = {
# # this allows \AA to be used in equations
# 'preamble': '\\global\\renewcommand{\\AA}{\\text{\\r{A}}}',
# }


# # Additional stuff for the LaTeX preamble.
# #latex_preamble = ''
LATEX_PREAMBLE = r"""
\newcommand{\lt}{<}
\newcommand{\gt}{>}
\renewcommand{\AA}{\text{\r{A}}} % Allow \AA in math mode
\usepackage[utf8]{inputenc}  % Allow unicode symbols in text
\DeclareUnicodeCharacter{00B7}{\ensuremath{\cdot}}   % cdot
\DeclareUnicodeCharacter{00B0}{\ensuremath{^\circ}}  % degrees
\DeclareUnicodeCharacter{212B}{\AA}                  % Angstrom
"""
latex_elements = {'preamble': LATEX_PREAMBLE}
