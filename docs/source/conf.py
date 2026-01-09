import os
import sys
from unittest.mock import MagicMock
import os
import sys
import shutil

# -- Project setup -----------------------------------------------------------

# AUTOMATIC COPY: Copy the root README.md to the docs folder
# This bypasses the "file outside source directory" error.
readme_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'README.md'))
readme_dst = os.path.abspath(os.path.join(os.path.dirname(__file__), 'README.md'))

if os.path.exists(readme_src):
#    shutil.copyfile(readme_src, readme_dst)
    print(f"Copied {readme_src} to {readme_dst}")
else:
    print(f"Warning: README.md not found at {readme_src}")

# ... rest of your conf.py configuration ...
# 1. Path setup: Points to your main project folder so it can find your code
sys.path.insert(0, os.path.abspath('../..')) # Use '../..' if conf.py is in docs/source
# OR use os.path.abspath('..') if conf.py is directly in docs/

# 2. Mocking: Prevents "ImportError: No module named PyQt5"
class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

MOCK_MODULES = [
    'PyQt5', 'PyQt5.QtCore', 'PyQt5.QtGui', 'PyQt5.QtWidgets',
    'vtk', 'vtk.qt.QVTKRenderWindowInteractor',
    'pyfftw', 'SimpleITK'
]
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MELAGE'
copyright = '2026, Bahram Jafrasteh'
author = 'Bahram Jafrasteh'
release = '2.0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_parser'
]


templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
