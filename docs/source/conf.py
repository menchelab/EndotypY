import os
import sys
sys.path.insert(0, os.path.abspath('../../'))  # Adjust to include your package

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
]
pygments_style = "sphinx"       # enable syntax highlighting

# Napoleon settings for Google/NumPy-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True

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

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_logo = "_static/logo.png"