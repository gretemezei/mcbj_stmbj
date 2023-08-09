# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__name__), '../..'))

import mcbj, pca_and_ml, plots, utils

project = 'MCBJ STM-BJ'
copyright = '2023, Greta Mezei'
author = 'Greta Mezei'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
   'sphinx.ext.autodoc',
   'sphinx.ext.autosummary',
   'sphinx.ext.mathjax',
   'sphinx.ext.viewcode',
   'sphinx.ext.intersphinx',
   'numpydoc',
]

intersphinx_mapping = {
   'python': ('https://docs.python.org/3.8/', None),
   'h5py': ('https://docs.h5py.org/en/stable/', None),
   'numpy': ('https://numpy.org/doc/stable/', None),
   'scipy': ('https://docs.scipy.org/doc/scipy/', None),
   'pandas': ('https://pandas.pydata.org/pandas-docs/stable', None)
}

default_role = 'py:obj'

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
