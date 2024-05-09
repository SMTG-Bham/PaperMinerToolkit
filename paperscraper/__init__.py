__version__ = '0.0.1'

from paperscraper.settings import update_elsevier_key, update_openai_key
import json
import os
import warnings

SETTINGS_FILE = os.path.join(os.path.expanduser('~'), '.config', '.pscraperrc.json')

def _load_config() -> dict[str, str]:
    try:
        with open(SETTINGS_FILE, mode='r', encoding="utf-8") as json_file:
            settings = json.load(json_file)
    except FileNotFoundError:
        update_elsevier_key({})
        update_openai_key({})
    except Exception as e:
        warnings.warn(f"Error loading {SETTINGS_FILE}: {e}.")

    return settings

SETTINGS = _load_config()
locals().update(SETTINGS)