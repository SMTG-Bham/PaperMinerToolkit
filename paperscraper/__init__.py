"""Expose package version and currently loaded PaperScraper settings."""

__version__ = '0.0.1'

from paperscraper.settings import (load_settings,
                                   update_elsevier_key,
                                   update_openai_key)

SETTINGS = load_settings()
locals().update(SETTINGS)
