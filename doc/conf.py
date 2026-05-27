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
# Include both class and __init__ docstrings
autoapi_python_class_content = "both"
# Group members by type (methods, attributes, etc.)
autoapi_member_order = "groupwise"
# Automatically add generated API docs to the topbar
autoapi_add_toctree_entry = True
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
# Enable the title of the home page when navigating with previous/next buttons
html_title = ""
html_theme_options = {
    # Maximum depth of the sidebar navigation tree
    "navigation_depth": 4,
    # GitHub icon in the top-right header
    "github_url": (
        "https://github.com/predictive-clinical-neuroscience/PCNtoolkit"
    ),
    # Do not show the "Edit this page" button on the right sidebar (links to
    # GitHub editor)
    "use_edit_page_button": False,
    # Clear the footer: remove copyright, Sphinx version,
    # and "Built with PyData Sphinx Theme" text
    "footer_start": [],
    "footer_end": [],
    # Version switcher shown in the top navbar, followed by
    # the dark/light toggle, then the GitHub icon link
    "navbar_end": [
        "version-switcher",
        "theme-switcher",
        "navbar-icon-links",
    ],
    "switcher": {
        # Stable URL so every deployed version can load the JSON list
        "json_url": (
            "https://pcntoolkit.readthedocs.io"
            "/en/stable/_static/switcher.json"
        ),
        # ReadTheDocs sets READTHEDOCS_VERSION automatically;
        # fall back to "dev" when building locally
        "version_match": os.environ.get(
            "READTHEDOCS_VERSION", "dev"
        ),
    },
    # Logo: icon image on the left, bold text on the right.
    # image_light / image_dark are relative to doc/ (the conf dir).
    "logo": {
        "text": "PCNtoolkit",
        "image_light": "_static/pcn-icon.png",
        "image_dark": "_static/pcn-icon.png",
    },
}
# Directory that holds static files (logo, custom CSS, etc.)
html_static_path = ["_static"]
# Apply custom CSS to add our own colours
html_css_files = ["custom.css"]
# Remove the "Show Source" link from the right sidebar (links to
# the raw .rst source file of that page)
html_show_sourcelink = False

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
