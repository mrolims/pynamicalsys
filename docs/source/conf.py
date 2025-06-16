import os
import sys


# This points to the src directory from docs/source
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)

from pynamicalsys import __version__

project = "pynamicalsys"
copyright = "2025, Matheus Rolim Sales and pynamicalsys authors"
author = "Matheus Rolim Sales"


release = __version__
if "dev" in __version__:
    version = __version__.split(".dev")[0] + "-dev"
else:
    version = ".".join(__version__.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.doctest",
    "myst_parser",
]

nb_execution_mode = "off"  # don't run notebooks during build

source_suffix = ".rst"

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"

# html_logo = "images/LOGO_v2.png"

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
html_static_path = ["_static"]

html_css_files = [
    "css/custom.css",
]
