"""Hold the PaperScraper version as a dependency-free constant.

The version lives in its own module so that packaging can read it by static
parse and so that low-level modules such as :mod:`paperscraper.provider` can
build a user agent from it without importing the package root, which would
pull in settings and the whole provider stack.
"""

from __future__ import annotations

__version__ = '0.0.1'
