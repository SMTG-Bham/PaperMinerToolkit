"""Expose package version and currently loaded PaperMiner settings."""

from paperminer._version import __version__ as __version__

from paperminer.settings import (load_settings as load_settings,
                                   update_elsevier_key as update_elsevier_key,
                                   update_openai_key as update_openai_key)

SETTINGS = load_settings()
locals().update(SETTINGS)
