"""Load, save, and update PaperScraper API and model configuration.

Settings are read from the user config file and environment variables. This
module also provides interactive command helpers for storing API keys and model
profiles used by search, download, and extraction workflows.
"""

import json
import openai
import os
import requests
from copy import deepcopy

from paperscraper import elsevier

SETTINGS_FILE = os.path.join(os.path.expanduser('~'), '.config', '.pscraperrc.json')
DEFAULT_MODEL = 'gpt-5.4-mini'
DEFAULT_TEMPERATURE = 0
DEFAULT_TOP_P = 1
DEFAULT_MODEL_PROFILE = {
    'provider': 'openai',
    'model': DEFAULT_MODEL,
    'base_url': None,
    'api_key': None,
    'capabilities': ['text'],
    'temperature': DEFAULT_TEMPERATURE,
    'top_p': DEFAULT_TOP_P,
}
DEFAULT_SETTINGS = {
    'model_profiles': {
        'text': DEFAULT_MODEL_PROFILE.copy(),
        'vision': {
            **DEFAULT_MODEL_PROFILE,
            'capabilities': ['text', 'vision'],
        },
    },
}


def _float_setting(value, default):
    """Convert an optional numeric setting to ``float`` with a default fallback."""
    if value is None or value == '':
        return default
    return float(value)


def _capabilities(value, default=None):
    """Normalize model capability settings into a list of capability names."""
    if value is None or value == '':
        return default or ['text']
    if isinstance(value, str):
        return [cap.strip() for cap in value.split(',') if cap.strip()]
    return value


def _merge_profile(default, configured):
    """Overlay a configured model profile on top of default profile values."""
    profile = default.copy()
    profile.update(configured or {})
    profile['capabilities'] = _capabilities(profile.get('capabilities'), default.get('capabilities'))
    profile['temperature'] = _float_setting(profile.get('temperature'), DEFAULT_TEMPERATURE)
    profile['top_p'] = _float_setting(profile.get('top_p'), DEFAULT_TOP_P)
    return profile


def _env_profile(prefix):
    """Collect model profile overrides from environment variables with ``prefix``."""
    env = {
        'provider': os.environ.get(f'{prefix}PROVIDER'),
        'model': os.environ.get(f'{prefix}NAME'),
        'base_url': os.environ.get(f'{prefix}BASE_URL'),
        'api_key': os.environ.get(f'{prefix}API_KEY'),
        'capabilities': os.environ.get(f'{prefix}CAPABILITIES'),
        'temperature': os.environ.get(f'{prefix}TEMPERATURE'),
        'top_p': os.environ.get(f'{prefix}TOP_P'),
    }
    return {key: value for key, value in env.items() if value is not None and value != ''}


def load_settings():
    """Load settings from disk and environment variables."""
    try:
        with open(SETTINGS_FILE, mode='r', encoding='utf-8') as json_file:
            settings = json.load(json_file)
    except FileNotFoundError:
        settings = {}
    except Exception as e:
        raise RuntimeError(f'Error loading {SETTINGS_FILE}: {e}.') from e

    merged = deepcopy(DEFAULT_SETTINGS)
    for key in ['elsevier_api_key', 'core_api_key', 'unpaywall_email', 'openai_api_key', 'anthropic_api_key']:
        if key in settings:
            merged[key] = settings[key]

    configured_profiles = settings.get('model_profiles', {})
    for profile_name, default_profile in DEFAULT_SETTINGS['model_profiles'].items():
        merged['model_profiles'][profile_name] = _merge_profile(default_profile, configured_profiles.get(profile_name))

    elsevier_api_key = os.environ.get('ELSEVIER_API_KEY')
    if elsevier_api_key:
        merged['elsevier_api_key'] = elsevier_api_key
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    if openai_api_key:
        merged['openai_api_key'] = openai_api_key
    anthropic_api_key = os.environ.get('ANTHROPIC_API_KEY')
    if anthropic_api_key:
        merged['anthropic_api_key'] = anthropic_api_key
    core_api_key = os.environ.get('CORE_API_KEY')
    if core_api_key:
        merged['core_api_key'] = core_api_key
    unpaywall_email = os.environ.get('UNPAYWALL_EMAIL')
    if unpaywall_email:
        merged['unpaywall_email'] = unpaywall_email

    text_env = _env_profile('PAPERSCRAPER_MODEL_')
    if text_env:
        current = merged['model_profiles']['text'].copy()
        current.update(text_env)
        merged['model_profiles']['text'] = _merge_profile(DEFAULT_SETTINGS['model_profiles']['text'], current)

    vision_env = _env_profile('PAPERSCRAPER_VISION_MODEL_')
    if vision_env:
        current = merged['model_profiles']['vision'].copy()
        current.update(vision_env)
        merged['model_profiles']['vision'] = _merge_profile(DEFAULT_SETTINGS['model_profiles']['vision'], current)
    return merged


def save_settings(settings):
    """Persist settings to the PaperScraper user config file."""
    os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
    with open(SETTINGS_FILE, mode='w', encoding='utf-8') as json_file:
        json.dump(settings, json_file, indent=2)


def get_model_profile(profile: str):
    """Return a named model profile after applying defaults and environment overrides."""
    settings = load_settings()
    try:
        return settings['model_profiles'][profile]
    except KeyError as e:
        raise KeyError(f'Model profile "{profile}" does not exist.') from e


def infer_model_capabilities(profile: str, model: str):
    """Infer whether a model profile should support text only or text plus vision."""
    model_lower = (model or '').lower()
    vision_markers = ['vl',
                      'vision',
                      'omni',
                      'llava',
                      'pixtral',
                      'molmo',
                      'internvl',
                      'qwen2.5-vl',
                      'qwen2-vl',
                      'qwen3-vl']
    if profile == 'vision' or any(marker in model_lower for marker in vision_markers):
        return ['text', 'vision']
    return ['text']


def set_model_profile(profile: str, provider: str, model: str, base_url=None, api_key=None, capabilities=None,
                      temperature=DEFAULT_TEMPERATURE, top_p=DEFAULT_TOP_P):
    """Store a model profile for later text or vision extraction runs."""
    settings = load_settings()
    capabilities = _capabilities(capabilities, infer_model_capabilities(profile, model))
    settings.setdefault('model_profiles', {})[profile] = {
        'provider': provider,
        'model': model,
        'base_url': base_url,
        'api_key': api_key,
        'capabilities': capabilities,
        'temperature': float(temperature),
        'top_p': float(top_p),
    }
    save_settings(settings)


def check_openai_api_key(api_key):
    """Return whether an OpenAI API key can authenticate against the models API."""
    client = openai.OpenAI(api_key=api_key)
    try:
        client.models.list()
    except openai.AuthenticationError:
        return False
    else:
        return True


def update_openai_key(settings=True):
    """Prompt for and save an OpenAI API key after validating it."""
    if settings:
        settings = load_settings()
    api_key = input('Enter OpenAI API key: ')
    if check_openai_api_key(api_key):
        settings['openai_api_key'] = api_key
        save_settings(settings)
    else:
        raise ValueError('OpenAI API key is invalid.')


def update_anthropic_key(settings=True):
    """Prompt for and save an Anthropic API key."""
    if settings:
        settings = load_settings()
    api_key = input('Enter Anthropic API key: ')
    settings['anthropic_api_key'] = api_key
    save_settings(settings)


def check_elsevier_api_key(api_key):
    """Return whether an Elsevier API key can run a minimal Scopus search."""
    try:
        url = elsevier.search_url('scopus', 'Test', 1, 'TITLE-ABS-KEY')
        elsevier.get_json(api_key, url)
    except requests.RequestException:
        return False
    else:
        return True


def update_elsevier_key(settings=True):
    """Prompt for, validate, and save an Elsevier API key."""
    if settings:
        settings = load_settings()
    api_key = input('Enter Elsevier API key: ')
    if check_elsevier_api_key(api_key):
        settings['elsevier_api_key'] = api_key
        save_settings(settings)
    else:
        raise ValueError('Elsevier API key is invalid.')


def update_core_key(settings=True):
    """Prompt for and save a CORE API key."""
    if settings:
        settings = load_settings()
    api_key = input('Enter CORE API key: ')
    settings['core_api_key'] = api_key
    save_settings(settings)


def update_unpaywall_email(settings=True):
    """Prompt for and save the email address sent to Unpaywall."""
    if settings:
        settings = load_settings()
    email = input('Enter Unpaywall email: ').strip()
    if '@' not in email:
        raise ValueError('Unpaywall email must be a valid email address.')
    settings['unpaywall_email'] = email
    save_settings(settings)


def update_model_settings(settings=True):
    """Interactively update one text or vision model profile."""
    if settings:
        settings = load_settings()
    profile = input('Enter model profile [text/vision]: ').strip() or 'text'
    provider = input('Enter model provider [openai/anthropic/local]: ').strip() or 'openai'
    model = input('Enter model name: ').strip()
    base_url = input('Enter base URL (leave blank for provider default): ').strip() or None
    api_key = input('Enter model API key (leave blank if not needed): ').strip() or None
    capabilities = input('Enter capabilities as comma-separated values [text,vision]: ').strip()
    temperature = input('Enter temperature [0]: ').strip() or DEFAULT_TEMPERATURE
    top_p = input('Enter top_p [1]: ').strip() or DEFAULT_TOP_P
    if not model:
        raise ValueError('Model name is required.')
    set_model_profile(profile, provider, model, base_url, api_key, capabilities or None, temperature=temperature,
                      top_p=top_p)
