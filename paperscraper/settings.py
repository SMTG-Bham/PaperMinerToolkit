from elsapy.elsclient import ElsClient
import openai
import requests
import json
import os

SETTINGS_FILE = os.path.join(os.path.expanduser('~'), '.config', '.pscraperrc.json')
DEFAULT_SETTINGS = {
    'model_provider': 'openai',
    'model_name': 'gpt-5-mini',
    'model_base_url': None,
    'model_api_key': None,
    'model_capabilities': ['text'],
}


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
    return merged


def save_settings(settings):
    os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
    with open(SETTINGS_FILE, mode='w', encoding='utf-8') as json_file:
        json.dump(settings, json_file, indent=2)


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
    provider = input('Enter model provider [openai/anthropic/openai-compatible/local]: ').strip() or 'openai'
    model = input('Enter model name: ').strip()
    base_url = input('Enter base URL (leave blank for provider default): ').strip() or None
    api_key = input('Enter model API key (leave blank if not needed): ').strip() or None
    capabilities = input('Enter capabilities as comma-separated values [text,vision]: ').strip()
    if not model:
        raise ValueError('Model name is required.')
    settings['model_provider'] = provider
    settings['model_name'] = model
    settings['model_base_url'] = base_url
    settings['model_api_key'] = api_key
    settings['model_capabilities'] = [cap.strip() for cap in capabilities.split(',') if cap.strip()] or ['text']
    save_settings(settings)
