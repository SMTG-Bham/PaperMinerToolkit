"""Load, save, and update PaperMiner API and model configuration.

Settings are read from the user config file and environment variables. This
module also provides interactive command helpers for storing API keys and model
profiles used by search, download, and extraction workflows.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import openai
import os
import requests
from copy import deepcopy
from typing import Any, Literal

SETTINGS_FILE = os.path.join(os.path.expanduser('~'), '.config', '.paperminerrc.json')
DEFAULT_MODEL = 'gpt-5.4-mini'
DEFAULT_TEMPERATURE = 0
DEFAULT_TOP_P = 1
DEFAULT_INPUT_TOKEN_LIMIT = 32000
DEFAULT_MODEL_PROFILE = {
    'provider': 'openai',
    'model': DEFAULT_MODEL,
    'base_url': None,
    'api_key': None,
    'capabilities': ['text'],
    'temperature': DEFAULT_TEMPERATURE,
    'top_p': DEFAULT_TOP_P,
    'input_token_limit': DEFAULT_INPUT_TOKEN_LIMIT,
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


def _float_setting(value: float | int | str | None, default: float) -> float:
    """Normalize an optional floating-point setting.

    Parameters
    ----------
    value : float | int | str | None
        Configured value. ``None`` and an empty string select the default.
    default : float
        Value returned when the configured value is missing.

    Returns
    -------
    float
        The default or the configured value converted to ``float``.

    Raises
    ------
    ValueError
        If a non-empty value cannot be converted to ``float``.
    """
    if value is None or value == '':
        return default
    return float(value)


def _capabilities(value: str | list[str] | None, default: list[str] | None = None) -> list[str]:
    """Normalize a model capability setting.

    Parameters
    ----------
    value : str | list[str] | None
        Comma-separated capabilities, an existing list, or a missing value.
    default : list[str] | None, optional
        Capabilities used when ``value`` is missing. Text-only capability is
        used when both inputs are missing.

    Returns
    -------
    list[str]
        Normalized capability names.
    """
    if value is None or value == '':
        return default or ['text']
    if isinstance(value, str):
        return [cap.strip() for cap in value.split(',') if cap.strip()]
    return value


def _merge_profile(
    default: Mapping[str, Any],
    configured: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Merge and normalize a configured model profile.

    Parameters
    ----------
    default : Mapping[str, Any]
        Default model profile values.
    configured : Mapping[str, Any] | None
        Stored or environment-derived overrides.

    Returns
    -------
    dict[str, Any]
        Merged profile with normalized capabilities, sampling values, and
        input token limit.
    """
    profile = default.copy()
    profile.update(configured or {})
    profile['capabilities'] = _capabilities(profile.get('capabilities'), default.get('capabilities'))
    profile['temperature'] = _float_setting(profile.get('temperature'), DEFAULT_TEMPERATURE)
    profile['top_p'] = _float_setting(profile.get('top_p'), DEFAULT_TOP_P)
    input_token_limit = profile.get('input_token_limit')
    if input_token_limit is None or input_token_limit == '':
        input_token_limit = DEFAULT_INPUT_TOKEN_LIMIT
    profile['input_token_limit'] = int(input_token_limit)
    return profile


def _env_profile(prefix: str) -> dict[str, str]:
    """Collect model profile overrides from environment variables.

    Parameters
    ----------
    prefix : str
        Environment variable prefix for the profile.

    Returns
    -------
    dict[str, str]
        Defined, non-empty environment overrides keyed by profile field.
    """
    env = {
        'provider': os.environ.get(f'{prefix}PROVIDER'),
        'model': os.environ.get(f'{prefix}NAME'),
        'base_url': os.environ.get(f'{prefix}BASE_URL'),
        'api_key': os.environ.get(f'{prefix}API_KEY'),
        'capabilities': os.environ.get(f'{prefix}CAPABILITIES'),
        'temperature': os.environ.get(f'{prefix}TEMPERATURE'),
        'top_p': os.environ.get(f'{prefix}TOP_P'),
        'input_token_limit': os.environ.get(f'{prefix}INPUT_TOKEN_LIMIT'),
    }
    return {key: value for key, value in env.items() if value is not None and value != ''}


def load_settings() -> dict[str, Any]:
    """Load effective PaperMiner settings.

    Values from the user config file are merged with package defaults, then
    overridden by supported environment variables.

    Returns
    -------
    dict[str, Any]
        Effective credentials and normalized text and vision model profiles.

    Raises
    ------
    RuntimeError
        If the settings file exists but cannot be read or decoded.
    ValueError
        If a configured numeric profile value is invalid.
    """
    try:
        with open(SETTINGS_FILE, mode='r', encoding='utf-8') as json_file:
            settings = json.load(json_file)
    except FileNotFoundError:
        settings = {}
    except Exception as e:
        raise RuntimeError(f'Error loading {SETTINGS_FILE}: {e}.') from e

    merged = deepcopy(DEFAULT_SETTINGS)
    for key in ['elsevier_api_key', 'core_api_key', 'unpaywall_email', 'openalex_api_key', 'openai_api_key',
                'anthropic_api_key', 'crossref_email', 'ncbi_api_key', 'ncbi_email']:
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
    openalex_api_key = os.environ.get('OPENALEX_API_KEY')
    if openalex_api_key:
        merged['openalex_api_key'] = openalex_api_key
    crossref_email = os.environ.get('CROSSREF_EMAIL')
    if crossref_email:
        merged['crossref_email'] = crossref_email
    ncbi_api_key = os.environ.get('NCBI_API_KEY')
    if ncbi_api_key:
        merged['ncbi_api_key'] = ncbi_api_key
    ncbi_email = os.environ.get('NCBI_EMAIL')
    if ncbi_email:
        merged['ncbi_email'] = ncbi_email

    text_env = _env_profile('PAPERMINER_MODEL_')
    if text_env:
        current = merged['model_profiles']['text'].copy()
        current.update(text_env)
        merged['model_profiles']['text'] = _merge_profile(DEFAULT_SETTINGS['model_profiles']['text'], current)

    vision_env = _env_profile('PAPERMINER_VISION_MODEL_')
    if vision_env:
        current = merged['model_profiles']['vision'].copy()
        current.update(vision_env)
        merged['model_profiles']['vision'] = _merge_profile(DEFAULT_SETTINGS['model_profiles']['vision'], current)
    return merged


def _save_settings(settings: dict[str, Any]) -> None:
    """Persist PaperMiner settings as JSON.

    Parameters
    ----------
    settings : dict[str, Any]
        Settings to write to the user config file.
    """
    os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
    with open(SETTINGS_FILE, mode='w', encoding='utf-8') as json_file:
        json.dump(settings, json_file, indent=2)


def get_model_profile(profile: str) -> dict[str, Any]:
    """Load an effective model profile by name.

    Parameters
    ----------
    profile : str
        Profile name, normally ``text`` or ``vision``.

    Returns
    -------
    dict[str, Any]
        Normalized model profile.

    Raises
    ------
    KeyError
        If the named profile does not exist.
    """
    settings = load_settings()
    try:
        return settings['model_profiles'][profile]
    except KeyError as e:
        raise KeyError(f'Model profile "{profile}" does not exist.') from e


def infer_model_capabilities(profile: str, model: str) -> list[str]:
    """Infer model capabilities from the profile and model name.

    Parameters
    ----------
    profile : str
        Model profile name.
    model : str
        Provider model identifier.

    Returns
    -------
    list[str]
        Text-only capability, or text and vision capabilities for a vision
        profile or vision-like model name.
    """
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


def set_model_profile(
    profile: str,
    provider: str,
    model: str,
    base_url: str | None = None,
    api_key: str | None = None,
    capabilities: str | list[str] | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    input_token_limit: int = DEFAULT_INPUT_TOKEN_LIMIT,
) -> None:
    """Store a model profile for extraction runs.

    Parameters
    ----------
    profile : str
        Profile name to create or replace.
    provider : str
        Model provider name.
    model : str
        Provider model identifier.
    base_url : str | None, optional
        Custom provider API base URL.
    api_key : str | None, optional
        Profile-specific provider credential.
    capabilities : str | list[str] | None, optional
        Explicit capabilities. Capabilities are inferred when omitted.
    temperature : float, default=DEFAULT_TEMPERATURE
        Sampling temperature.
    top_p : float, default=DEFAULT_TOP_P
        Nucleus sampling probability mass.
    input_token_limit : int, default=DEFAULT_INPUT_TOKEN_LIMIT
        Maximum model input tokens before chunking.
    """
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
        'input_token_limit': int(input_token_limit),
    }
    _save_settings(settings)


def _check_openai_api_key(api_key: str) -> bool:
    """Validate an OpenAI API key against the models API.

    Parameters
    ----------
    api_key : str
        OpenAI API key to validate.

    Returns
    -------
    bool
        ``False`` for an authentication failure; ``True`` after a successful
        model listing.
    """
    client = openai.OpenAI(api_key=api_key)
    try:
        client.models.list()
    except openai.AuthenticationError:
        return False
    else:
        return True


def _mask_secret(value: object) -> str:
    """Mask a configured secret for display.

    Parameters
    ----------
    value : object
        Secret value, or a false value when no secret is configured.

    Returns
    -------
    str
        ``not set``, a fully masked short value, or the first and last four
        characters of a longer value.
    """
    if not value:
        return 'not set'
    value = str(value)
    if len(value) <= 8:
        return '********'
    return f'{value[:4]}...{value[-4:]}'


def _show_current_setting(
    settings: Mapping[str, Any],
    key: str,
    label: str,
    secret: bool = True,
) -> None:
    """Print a configured value before prompting for its replacement.

    Parameters
    ----------
    settings : Mapping[str, Any]
        Settings containing the value to display.
    key : str
        Settings key to read.
    label : str
        Human-readable label included in the prompt.
    secret : bool, default=True
        Whether to mask the configured value.
    """
    value = settings.get(key)
    if secret:
        value = _mask_secret(value)
    elif not value:
        value = 'not set'
    print(f'Current {label}: {value}')


def update_openai_key(settings: dict[str, Any] | Literal[True] = True) -> None:
    """Prompt for, validate, and save an OpenAI API key.

    Parameters
    ----------
    settings : dict[str, Any] or Literal[True], default=True
        Settings mapping to update. A true value loads the current settings.

    Raises
    ------
    ValueError
        If the entered key fails OpenAI authentication.
    """
    if settings:
        settings = load_settings()
    _show_current_setting(settings, 'openai_api_key', 'OpenAI API key')
    api_key = input('Enter OpenAI API key: ')
    if _check_openai_api_key(api_key):
        settings['openai_api_key'] = api_key
        _save_settings(settings)
    else:
        raise ValueError('OpenAI API key is invalid.')


def update_anthropic_key(settings: dict[str, Any] | Literal[True] = True) -> None:
    """Prompt for and save an Anthropic API key.

    Parameters
    ----------
    settings : dict[str, Any] or Literal[True], default=True
        Settings mapping to update. A true value loads the current settings.
    """
    if settings:
        settings = load_settings()
    _show_current_setting(settings, 'anthropic_api_key', 'Anthropic API key')
    api_key = input('Enter Anthropic API key: ')
    settings['anthropic_api_key'] = api_key
    _save_settings(settings)


def _check_elsevier_api_key(api_key: str) -> bool:
    """Validate an Elsevier API key with a minimal Scopus search.

    Parameters
    ----------
    api_key : str
        Elsevier API key to validate.

    Returns
    -------
    bool
        ``True`` when the request succeeds, otherwise ``False``.
    """
    from paperminer.elsevier import check_api_key
    return check_api_key(api_key)


def update_elsevier_key(settings: dict[str, Any] | Literal[True] = True) -> None:
    """Prompt for, validate, and save an Elsevier API key.

    Parameters
    ----------
    settings : dict[str, Any] or Literal[True], default=True
        Settings mapping to update. A true value loads the current settings.

    Raises
    ------
    ValueError
        If the entered key fails Elsevier validation.
    """
    if settings:
        settings = load_settings()
    _show_current_setting(settings, 'elsevier_api_key', 'Elsevier API key')
    api_key = input('Enter Elsevier API key: ')
    if _check_elsevier_api_key(api_key):
        settings['elsevier_api_key'] = api_key
        _save_settings(settings)
    else:
        raise ValueError('Elsevier API key is invalid.')


def update_core_key(settings: dict[str, Any] | Literal[True] = True) -> None:
    """Prompt for and save a CORE API key.

    Parameters
    ----------
    settings : dict[str, Any] or Literal[True], default=True
        Settings mapping to update. A true value loads the current settings.
    """
    if settings:
        settings = load_settings()
    _show_current_setting(settings, 'core_api_key', 'CORE API key')
    api_key = input('Enter CORE API key: ')
    settings['core_api_key'] = api_key
    _save_settings(settings)


def update_unpaywall_email(settings: dict[str, Any] | Literal[True] = True) -> None:
    """Prompt for and save the email address sent to Unpaywall.

    Parameters
    ----------
    settings : dict[str, Any] or Literal[True], default=True
        Settings mapping to update. A true value loads the current settings.

    Raises
    ------
    ValueError
        If the entered value does not contain an ``@`` character.
    """
    if settings:
        settings = load_settings()
    _show_current_setting(settings, 'unpaywall_email', 'Unpaywall email', secret=False)
    email = input('Enter Unpaywall email: ').strip()
    if '@' not in email:
        raise ValueError('Unpaywall email must be a valid email address.')
    settings['unpaywall_email'] = email
    _save_settings(settings)


def update_crossref_email(settings: dict[str, Any] | Literal[True] = True) -> None:
    """Prompt for and save the contact email sent to Crossref.

    Crossref asks that automated clients identify themselves with a contact
    address. The same address is reused as the OpenAlex ``mailto`` parameter.

    Parameters
    ----------
    settings : dict[str, Any] or Literal[True], default=True
        Settings mapping to update. A true value loads the current settings.

    Raises
    ------
    ValueError
        If the entered value does not contain an ``@`` character.
    """
    if settings:
        settings = load_settings()
    _show_current_setting(settings, 'crossref_email', 'Crossref email', secret=False)
    email = input('Enter Crossref email: ').strip()
    if '@' not in email:
        raise ValueError('Crossref email must be a valid email address.')
    settings['crossref_email'] = email
    _save_settings(settings)


def update_ncbi_key(settings: dict[str, Any] | Literal[True] = True) -> None:
    """Prompt for and save an NCBI E-utilities API key.

    The key is optional. PubMed and PMC serve unauthenticated clients at three
    requests per second; a key raises that ceiling to ten.

    Parameters
    ----------
    settings : dict[str, Any] or Literal[True], default=True
        Settings mapping to update. A true value loads the current settings.
    """
    if settings:
        settings = load_settings()
    _show_current_setting(settings, 'ncbi_api_key', 'NCBI API key')
    api_key = input('Enter NCBI API key: ')
    settings['ncbi_api_key'] = api_key
    _save_settings(settings)


def update_ncbi_email(settings: dict[str, Any] | Literal[True] = True) -> None:
    """Prompt for and save the contact email sent to NCBI E-utilities.

    NCBI asks that automated clients identify themselves so it can warn a
    contact before blocking an address. PaperMiner falls back to the Crossref
    address when no NCBI-specific address is configured.

    Parameters
    ----------
    settings : dict[str, Any] or Literal[True], default=True
        Settings mapping to update. A true value loads the current settings.

    Raises
    ------
    ValueError
        If the entered value does not contain an ``@`` character.
    """
    if settings:
        settings = load_settings()
    _show_current_setting(settings, 'ncbi_email', 'NCBI email', secret=False)
    email = input('Enter NCBI email: ').strip()
    if '@' not in email:
        raise ValueError('NCBI email must be a valid email address.')
    settings['ncbi_email'] = email
    _save_settings(settings)


def _check_openalex_api_key(api_key: str) -> bool:
    """Check whether OpenAlex explicitly rejects an API key.

    Parameters
    ----------
    api_key : str
        OpenAlex API key to validate.

    Returns
    -------
    bool
        ``False`` only for an HTTP 401 response. Other statuses and connection
        failures return ``True`` so transient outages do not block saving a
        key.
    """
    from paperminer import openalex

    try:
        response = requests.get(openalex.RATE_LIMIT_URL,
                                params={'api_key': api_key},
                                headers=openalex.request_headers(),
                                timeout=30)
    except requests.RequestException:
        return True
    return response.status_code != 401


def update_openalex_key(settings: dict[str, Any] | Literal[True] = True) -> None:
    """Prompt for, validate, and save an OpenAlex API key.

    OpenAlex permits unauthenticated requests with a smaller daily credit
    budget; a free API key increases that budget.

    Parameters
    ----------
    settings : dict[str, Any] or Literal[True], default=True
        Settings mapping to update. A true value loads the current settings.

    Raises
    ------
    ValueError
        If OpenAlex explicitly rejects the entered key.
    """
    if settings:
        settings = load_settings()
    _show_current_setting(settings, 'openalex_api_key', 'OpenAlex API key')
    api_key = input('Enter OpenAlex API key: ').strip()
    if _check_openalex_api_key(api_key):
        settings['openalex_api_key'] = api_key
        _save_settings(settings)
    else:
        raise ValueError('OpenAlex API key is invalid.')
