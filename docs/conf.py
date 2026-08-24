"""Sphinx configuration for the PaperMiner documentation."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

project = "PaperMiner"
author = "PaperMiner contributors"
copyright = "2026, PaperMiner contributors"

try:
    release = version("paperminer")
except PackageNotFoundError:
    release = "0.0.1"
version = release

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_click",
    "sphinxcontrib.mermaid",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "substitution",
    "tasklist",
]
nb_execution_mode = "off"
nb_merge_streams = True

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_mock_imports = [
    "fitz",
    "headroom",
    "joblib",
    "matplotlib",
    "numpy",
    "openai",
    "pandas",
    "PIL",
    "pypdf",
    "regex",
    "requests",
    "scipy",
    "sklearn",
    "tiktoken",
    "tqdm",
    "transformers",
]
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static", "../assets"]
html_css_files = ["custom.css"]
html_title = f"PaperMiner {release}"
html_favicon = "../assets/Paper_Miner.svg"
html_theme_options = {
    "logo": {
        "image_light": "Paper_Miner_banner_large_text_light.svg",
        "image_dark": "Paper_Miner_banner_large_text_dark.svg",
        "alt_text": "PaperMiner",
    },
    "header_links_before_dropdown": 6,
    "show_toc_level": 2,
    "use_edit_page_button": True,
    "pygments_light_style": "tango",
    "pygments_dark_style": "monokai",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/SMTG-Bham/PaperMiner",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        }
    ],
}
html_context = {
    "github_user": "SMTG-Bham",
    "github_repo": "PaperMiner",
    "github_version": "main",
    "doc_path": "docs",
}

mermaid_version = "11.4.1"
