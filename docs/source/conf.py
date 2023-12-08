import sys
from pathlib import Path

src_path = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_path))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Time Series Toolkit"
copyright = "PR Lab"
author = "Hassan Ahmed"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "nbsphinx"]

source_suffix = [".rst", ".md"]
nbsphinx_execute = "never"

templates_path = ["_templates"]
exclude_patterns = []  # ['tests/*', 'utils.py']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "classic"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "collapsiblesidebar": "false",
    "stickysidebar": "true",
    "globaltoc_collapse": "true",
    "body_min_width": "70%",
    "sidebarwidth": 450,
    "relbarbgcolor": "#009899",
    "bodyfont": "Courier New",
}
html_short_title: str = "Time Series Toolkit"
