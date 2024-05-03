__version__ = '0.0.1'

import json
import os
import warnings

SETTINGS_FILE = os.path.join(os.path.expanduser('~'), '.config', '.pscraperrc.json')

def _load_config() -> dict[str, str]:
    # if file doesnt exist, create file and ask for api keys
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as json_file:
            settings = json.load(json_file) or {}
        pass
    except FileNotFoundError:
        pass
    except Exception as exc:
        warnings.warn(f"Error loading {SETTINGS_FILE}: {exc}.")

    return settings

SETTINGS = _load_config()
locals().update(SETTINGS)