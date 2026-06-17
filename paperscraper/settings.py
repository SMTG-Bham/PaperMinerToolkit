from elsapy.elsclient import ElsClient
import openai
import requests
import json
import os

SETTINGS_FILE = os.path.join(os.path.expanduser('~'), '.config', '.pscraperrc.json')
DEFAULT_MODEL_PROFILE = {
    'provider': 'openai',
    'model': 'gpt-5-mini',
    'base_url': None,
    'api_key': None,
    'capabilities': ['text'],
}
DEFAULT_SETTINGS = {
    'model_provider': 'openai',
    'model_name': 'gpt-5-mini',
    'model_base_url': None,
    'model_api_key': None,
    'model_capabilities': ['text'],
    'model_profiles': {
        'text': DEFAULT_MODEL_PROFILE.copy(),
        'vision': {
            'provider': 'openai',
            'model': 'gpt-5-mini',
            'base_url': None,
            'api_key': None,
            'capabilities': ['text', 'vision'],
        },
    },
}


def _legacy_profile(settings):
    return {
        'provider': settings.get('model_provider') or 'openai',
        'model': settings.get('model_name') or 'gpt-5-mini',
        'base_url': settings.get('model_base_url'),
        'api_key': settings.get('model_api_key'),
        'capabilities': settings.get('model_capabilities') or ['text'],
    }


def _merge_profile(default, configured):
    profile = default.copy()
    profile.update(configured or {})
    capabilities = profile.get('capabilities') or ['text']
    if isinstance(capabilities, str):
        capabilities = [cap.strip() for cap in capabilities.split(',') if cap.strip()]
    profile['capabilities'] = capabilities
    return profile


def load_settings():
    try:
        with open(SETTINGS_FILE, mode='r', encoding='utf-8') as json_file:
            settings = json.load(json_file)
    except FileNotFoundError:
        settings = {}
    except Exception as e:
        raise RuntimeError(f'Error loading {SETTINGS_FILE}: {e}.') from e

    merged = DEFAULT_SETTINGS.copy()
    merged.update(settings)
    profiles = DEFAULT_SETTINGS['model_profiles'].copy()
    profiles.update(settings.get('model_profiles', {}))
    profiles['text'] = _merge_profile(_legacy_profile(merged), profiles.get('text'))
    profiles['vision'] = _merge_profile(DEFAULT_SETTINGS['model_profiles']['vision'], profiles.get('vision'))
    merged['model_profiles'] = profiles

    env_overrides = {
        'elsevier_api_key': os.environ.get('ELSEVIER_API_KEY'),
        'openai_api_key': os.environ.get('OPENAI_API_KEY'),
        'model_provider': os.environ.get('PAPERSCRAPER_MODEL_PROVIDER'),
        'model_name': os.environ.get('PAPERSCRAPER_MODEL_NAME'),
        'model_base_url': os.environ.get('PAPERSCRAPER_MODEL_BASE_URL'),
        'model_api_key': os.environ.get('PAPERSCRAPER_MODEL_API_KEY'),
    }
    for key, value in env_overrides.items():
        if value:
            merged[key] = value
    capabilities = os.environ.get('PAPERSCRAPER_MODEL_CAPABILITIES')
    if capabilities:
        merged['model_capabilities'] = [cap.strip() for cap in capabilities.split(',') if cap.strip()]

    # Environment variables target the text profile by default, matching existing HPC usage.
    if any(merged.get(key) for key in ['model_provider', 'model_name', 'model_base_url', 'model_api_key']):
        merged['model_profiles']['text'] = _legacy_profile(merged)
    vision_prefix = 'PAPERSCRAPER_VISION_MODEL_'
    vision_env = {
        'provider': os.environ.get(f'{vision_prefix}PROVIDER'),
        'model': os.environ.get(f'{vision_prefix}NAME'),
        'base_url': os.environ.get(f'{vision_prefix}BASE_URL'),
        'api_key': os.environ.get(f'{vision_prefix}API_KEY'),
        'capabilities': os.environ.get(f'{vision_prefix}CAPABILITIES'),
    }
    if any(vision_env.values()):
        current = merged['model_profiles']['vision'].copy()
        for key, value in vision_env.items():
            if value:
                current[key] = value
        merged['model_profiles']['vision'] = _merge_profile(DEFAULT_SETTINGS['model_profiles']['vision'], current)
    return merged


def save_settings(settings):
    os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
    with open(SETTINGS_FILE, mode='w', encoding='utf-8') as json_file:
        json.dump(settings, json_file, indent=2)


def get_model_profile(profile: str):
    settings = load_settings()
    try:
        return settings['model_profiles'][profile]
    except KeyError as e:
        raise KeyError(f'Model profile "{profile}" does not exist.') from e


def infer_model_capabilities(profile: str, model: str):
    model_lower = (model or '').lower()
    vision_markers = ['vl', 'vision', 'omni', 'llava', 'pixtral', 'molmo', 'internvl', 'qwen2.5-vl', 'qwen2-vl', 'qwen3-vl']
    if profile == 'vision' or any(marker in model_lower for marker in vision_markers):
        return ['text', 'vision']
    return ['text']


def set_model_profile(profile: str, provider: str, model: str, base_url=None, api_key=None, capabilities=None):
    settings = load_settings()
    capabilities = capabilities or infer_model_capabilities(profile, model)
    if isinstance(capabilities, str):
        capabilities = [cap.strip() for cap in capabilities.split(',') if cap.strip()]
    settings.setdefault('model_profiles', {})[profile] = {
        'provider': provider,
        'model': model,
        'base_url': base_url,
        'api_key': api_key,
        'capabilities': capabilities,
    }
    if profile == 'text':
        settings['model_provider'] = provider
        settings['model_name'] = model
        settings['model_base_url'] = base_url
        settings['model_api_key'] = api_key
        settings['model_capabilities'] = capabilities
    save_settings(settings)


def check_openai_api_key(api_key):
    client = openai.OpenAI(api_key=api_key)
    try:
        client.models.list()
    except openai.AuthenticationError:
        return False
    else:
        return True


def update_openai_key(settings=True):
    if settings:
        settings = load_settings()
    api_key = input('Enter OpenAI API key: ')
    if check_openai_api_key(api_key):
        settings['openai_api_key'] = api_key
        save_settings(settings)
    else:
        raise ValueError('OpenAI API key is invalid.')


def check_elsevier_api_key(api_key):
    client = ElsClient(api_key)
    try:
        url = 'https://api.elsevier.com/content/search/scopus?query=Test&count=1'
        client.exec_request(url)
    except requests.HTTPError:
        return False
    else:
        return True


def update_elsevier_key(settings=True):
    if settings:
        settings = load_settings()
    api_key = input('Enter Elsevier API key: ')
    if check_elsevier_api_key(api_key):
        settings['elsevier_api_key'] = api_key
        save_settings(settings)
    else:
        raise ValueError('Elsevier API key is invalid.')


def update_model_settings(settings=True):
    if settings:
        settings = load_settings()
    profile = input('Enter model profile [text/vision]: ').strip() or 'text'
    provider = input('Enter model provider [openai/anthropic/openai-compatible/local/hpc]: ').strip() or 'openai'
    model = input('Enter model name: ').strip()
    base_url = input('Enter base URL (leave blank for provider default): ').strip() or None
    api_key = input('Enter model API key (leave blank if not needed): ').strip() or None
    capabilities = input('Enter capabilities as comma-separated values [text,vision]: ').strip()
    if not model:
        raise ValueError('Model name is required.')
    set_model_profile(profile, provider, model, base_url, api_key, capabilities or None)
