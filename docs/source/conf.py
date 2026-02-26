import os
import sys
sys.path.insert(0, os.path.abspath('../../')) # Points 2 levels up to your code

extensions = [
    'sphinx.ext.autodoc',     # Pulls docstrings from code
    'sphinx.ext.viewcode',    # Adds "source" links to your docs
    'sphinx.ext.napoleon',    # Support for Google/NumPy style docstrings
]

html_theme = 'sphinx_rtd_theme' # This is the classic "Read the Docs" look
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'EndotypY'
copyright = '2026, Chloé Bûcheron, Iker Núñez-Carpintero, Mathilde Meyenberg, Enes Sakalli'
author = 'Chloé Bûcheron, Iker Núñez-Carpintero, Mathilde Meyenberg, Enes Sakalli'
release = '0.0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
