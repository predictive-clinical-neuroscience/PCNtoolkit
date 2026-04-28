# Configuration file for the Sphinx documentation builder
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# Project information
project = "PCNToolkit"
copyright = "2025, Andre Marquand"
author = "Andre Marquand"
release = "1.0.0"

# Extensions
extensions = [
    "autoapi.extension",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
]

# AutoAPI settings
autoapi_dirs = ["../pcntoolkit"]  # Directory to scan
autoapi_options = [
    "members",  # Include class/module members
    "undoc-members",  # Include items without docstrings
    "show-inheritance",  # Show base classes
    "show-module-summary",  # Include module docstring summaries
    "special-members",  # Include special methods (__init__, etc.)
]
autoapi_python_class_content = "both"  # Include both class and __init__ docstrings
autoapi_member_order = "groupwise"  # Group members by type (methods, attributes, etc.)
# Disable automatic toctree injection — we place the API
# Reference section manually in developers/index.rst
autoapi_add_toctree_entry = False
autoapi_template_dir = "_templates/autoapi"  # Custom templates location
autoapi_keep_files = True  # Keep generated RST files for debugging

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# Theme settings
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    # Maximum depth of the sidebar navigation tree
    "navigation_depth": 4,
    # GitHub icon in the top-right header
    "github_url": (
        "https://github.com/predictive-clinical-neuroscience/PCNtoolkit"
    ),
    # Do not show the "Edit this page" button
    "use_edit_page_button": False,
    # Logo shown in the top-left of every page
    "logo": {
        "image_light": "pcn-logo.png",
        "image_dark": "pcn-logo.png",
    },
}
# Directory that holds static files (logo, custom CSS, etc.)
html_static_path = ["_static"]
# Apply custom CSS to hide breadcrumbs and other overrides
html_css_files = ["custom.css"]

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# General settings
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
add_module_names = False
nitpicky = True

# AutoDoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "imported-members": True,
}


def run_notebook_conversion(app):
    pass
    # import pathlib
    # import subprocess
    # script_path = pathlib.Path(__file__).parent / 'convert_notebooks.py'
    # subprocess.run([sys.executable, str(script_path)], check=True)


def setup(app):
    app.connect("builder-inited", run_notebook_conversion)
