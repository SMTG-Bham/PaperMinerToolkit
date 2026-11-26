"""Load, save, and update PaperMinerToolkit API and model configuration.

Settings are read from the user config file and environment variables. This
module also provides interactive command helpers for storing API keys and model
profiles used by search, download, and extraction workflows.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import openai
import os
import requests
from copy import deepcopy
from typing import Any, Literal

SETTINGS_FILE = os.path.join(os.path.expanduser('~'), '.config', '.paperminertoolkitrc.json')
DEFAULT_MODEL = 'gpt-5.4-mini'
DEFAULT_TEMPERATURE = 0
DEFAULT_TOP_P = 1
DEFAULT_INPUT_TOKEN_LIMIT = 32000
DEFAULT_OUTPUT_TOKEN_LIMIT = 10000
DEFAULT_REQUEST_TIMEOUT = 600
DEFAULT_MODEL_PROFILE = {
    'provider': 'openai',
    'model': DEFAULT_MODEL,
    'base_url': None,
    'api_key': None,
    'capabilities': ['text'],
    'temperature': DEFAULT_TEMPERATURE,
    'top_p': DEFAULT_TOP_P,
    'input_token_limit': DEFAULT_INPUT_TOKEN_LIMIT,
    'output_token_limit': DEFAULT_OUTPUT_TOKEN_LIMIT,
    'request_timeout': DEFAULT_REQUEST_TIMEOUT,
    'reasoning_effort': None,
    'thinking': None,
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


# These keys are accepted from both the settings file and the environment.
SETTING_ENV_VARS = {
    'elsevier_api_key': 'ELSEVIER_API_KEY',
    'core_api_key': 'CORE_API_KEY',
    'core_min_interval': 'CORE_MIN_INTERVAL',
    'unpaywall_email': 'UNPAYWALL_EMAIL',
    'openalex_api_key': 'OPENALEX_API_KEY',
    'openai_api_key': 'OPENAI_API_KEY',
    'anthropic_api_key': 'ANTHROPIC_API_KEY',
    'crossref_email': 'CROSSREF_EMAIL',
    'ncbi_api_key': 'NCBI_API_KEY',
    'ncbi_email': 'NCBI_EMAIL',
}
MODEL_ENV_PREFIXES = {
    'text': 'PAPERMINERTOOLKIT_MODEL_',
    'vision': 'PAPERMINERTOOLKIT_VISION_MODEL_',
}
MODEL_ENV_FIELDS = {
    'provider': 'PROVIDER',
    'model': 'NAME',
    'base_url': 'BASE_URL',
    'api_key': 'API_KEY',
    'capabilities': 'CAPABILITIES',
    'temperature': 'TEMPERATURE',
    'top_p': 'TOP_P',
    'input_token_limit': 'INPUT_TOKEN_LIMIT',
    'output_token_limit': 'OUTPUT_TOKEN_LIMIT',
    'request_timeout': 'REQUEST_TIMEOUT',
    'reasoning_effort': 'REASONING_EFFORT',
    'thinking': 'THINKING',
}
_THINKING_VALUES = {
    'true': True, '1': True, 'yes': True, 'on': True,
    'false': False, '0': False, 'no': False, 'off': False,
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


def reasoning_effort_setting(value: str | None) -> str | None:
    """Normalize an optional reasoning-effort setting.

    Parameters
    ----------
    value : str | None
        Configured effort, such as ``"low"`` or ``"high"``. Levels differ
        between models, so any non-empty value is accepted.

    Returns
    -------
    str | None
        The stripped effort, or ``None`` (use the model's default) when the
        value is missing or blank.
    """
    if value is None:
        return None
    return str(value).strip() or None


def thinking_setting(value: bool | str | None) -> bool | None:
    """Normalize an optional thinking on/off setting.

    Parameters
    ----------
    value : bool | str | None
        ``True``/``False`` or a string such as ``"on"``, ``"off"``,
        ``"true"`` or ``"0"``.

    Returns
    -------
    bool | None
        Whether thinking is switched on, or ``None`` (use the model's
        default) when the value is missing or blank.

    Raises
    ------
    ValueError
        If a string is not a recognised on/off value.
    """
    if value is None or isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return None
    if text not in _THINKING_VALUES:
        raise ValueError(f'thinking must be on or off, not {value!r}')
    return _THINKING_VALUES[text]


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
        input and output token limits, request timeout and reasoning settings.
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
    output_token_limit = profile.get('output_token_limit')
    if output_token_limit is None or output_token_limit == '':
        output_token_limit = DEFAULT_OUTPUT_TOKEN_LIMIT
    profile['output_token_limit'] = int(output_token_limit)
    request_timeout = profile.get('request_timeout')
    if request_timeout is None or request_timeout == '':
        request_timeout = DEFAULT_REQUEST_TIMEOUT
    profile['request_timeout'] = float(request_timeout)
    profile['reasoning_effort'] = reasoning_effort_setting(profile.get('reasoning_effort'))
    profile['thinking'] = thinking_setting(profile.get('thinking'))
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
        key: os.environ.get(f'{prefix}{suffix}')
        for key, suffix in MODEL_ENV_FIELDS.items()
    }
    return {key: value for key, value in env.items() if value is not None and value != ''}


def load_settings() -> dict[str, Any]:
    """Load effective PaperMinerToolkit settings.

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
    for key in SETTING_ENV_VARS:
        if key in settings:
            merged[key] = settings[key]

    configured_profiles = settings.get('model_profiles', {})
    for profile_name, default_profile in DEFAULT_SETTINGS['model_profiles'].items():
        merged['model_profiles'][profile_name] = _merge_profile(default_profile, configured_profiles.get(profile_name))

    for key, env_name in SETTING_ENV_VARS.items():
        value = os.environ.get(env_name)
        if value:
            merged[key] = value

    for profile_name, prefix in MODEL_ENV_PREFIXES.items():
        overrides = _env_profile(prefix)
        if overrides:
            current = merged['model_profiles'][profile_name].copy()
            current.update(overrides)
            merged['model_profiles'][profile_name] = _merge_profile(
                DEFAULT_SETTINGS['model_profiles'][profile_name], current
            )
    return merged


def _saved_settings() -> dict[str, Any]:
    """Read only what the settings file holds, without defaults or environment overrides.

    Setters update this and write it back, so a credential supplied through an
    environment variable is never copied into the settings file.

    Returns
    -------
    dict[str, Any]
        The decoded settings file, or an empty mapping when it does not exist.

    Raises
    ------
    RuntimeError
        If the settings file exists but cannot be read or decoded.
    """
    try:
        with open(SETTINGS_FILE, mode='r', encoding='utf-8') as json_file:
            saved = json.load(json_file)
    except FileNotFoundError:
        return {}
    except Exception as e:
        raise RuntimeError(f'Error loading {SETTINGS_FILE}: {e}.') from e
    if not isinstance(saved, dict):
        raise RuntimeError(f'Error loading {SETTINGS_FILE}: expected a JSON object.')
    return saved


def _save_settings(settings: dict[str, Any]) -> None:
    """Persist PaperMinerToolkit settings as JSON.

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
    output_token_limit: int = DEFAULT_OUTPUT_TOKEN_LIMIT,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    reasoning_effort: str | None = None,
    thinking: bool | None = None,
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
    output_token_limit : int, default=DEFAULT_OUTPUT_TOKEN_LIMIT
        Maximum tokens the model may generate per extraction request,
        including any reasoning a server counts against it.
    request_timeout : float, default=DEFAULT_REQUEST_TIMEOUT
        Seconds to wait for one request to an OpenAI-compatible server.
    reasoning_effort : str | None, optional
        Reasoning effort sent with each request (e.g. ``"low"``, ``"medium"``,
        ``"high"``). ``None`` leaves the model's default.
    thinking : bool | None, optional
        Switch a hybrid model's thinking on or off through its chat template
        (OpenAI-compatible servers only). ``None`` leaves the model's default.
    """
    settings = _saved_settings()
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
        'output_token_limit': int(output_token_limit),
        'request_timeout': float(request_timeout),
        'reasoning_effort': reasoning_effort_setting(reasoning_effort),
        'thinking': thinking_setting(thinking),
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


def _update_api_key(
    settings: dict[str, Any] | Literal[True],
    key: str,
    label: str,
    validate: Callable[[str], bool] | None = None,
    *,
    strip: bool = False,
    value: str | None = None,
) -> None:
    """Prompt for, optionally validate, and persist one provider API key.

    Parameters
    ----------
    settings : dict[str, Any] or Literal[True]
        Mapping to update in place, or ``True`` to load the current settings.
    key : str
        Settings key containing the credential.
    label : str
        Label used in the display, prompt, and validation error.
    validate : callable or None, optional
        Provider-specific check, if the provider supports key validation.
    strip : bool, default=False
        Whether this provider trims surrounding whitespace from entered keys.
    value : str or None, optional
        Key to save without prompting; ``None`` prompts on stdin.

    Raises
    ------
    ValueError
        If the provider rejects the entered key.
    """
    if settings is True:
        settings = _saved_settings()
    if value is None:
        _show_current_setting(settings, key, label)
        api_key = input(f'Enter {label}: ')
    else:
        api_key = value
    if strip:
        api_key = api_key.strip()
    if validate is not None and not validate(api_key):
        raise ValueError(f'{label} is invalid.')
    settings[key] = api_key
    _save_settings(settings)


def update_openai_key(settings: dict[str, Any] | Literal[True] = True,
                      value: str | None = None) -> None:
    """Prompt for, validate, and save an OpenAI API key.

    Parameters
    ----------
    settings : dict[str, Any] or Literal[True], default=True
        Mapping to update in place, or ``True`` to load the current settings.
    value : str or None, optional
        Value to save without prompting; ``None`` prompts on stdin.

    Raises
    ------
    ValueError
        If the entered key fails OpenAI authentication.
    """
    _update_api_key(settings, 'openai_api_key', 'OpenAI API key', _check_openai_api_key, value=value)


def update_anthropic_key(settings: dict[str, Any] | Literal[True] = True,
                         value: str | None = None) -> None:
    """Prompt for and save an Anthropic API key.

    Parameters
    ----------
    settings : dict[str, Any] or Literal[True], default=True
        Mapping to update in place, or ``True`` to load the current settings.
    value : str or None, optional
        Value to save without prompting; ``None`` prompts on stdin.
    """
    _update_api_key(settings, 'anthropic_api_key', 'Anthropic API key', value=value)


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
    from paperminertoolkit.providers.elsevier import check_api_key
    return check_api_key(api_key)


def update_elsevier_key(settings: dict[str, Any] | Literal[True] = True,
                        value: str | None = None) -> None:
    """Prompt for, validate, and save an Elsevier API key.

    Parameters
    ----------
    settings : dict[str, Any] or Literal[True], default=True
        Mapping to update in place, or ``True`` to load the current settings.
    value : str or None, optional
        Value to save without prompting; ``None`` prompts on stdin.

    Raises
    ------
    ValueError
        If the entered key fails Elsevier validation.
    """
    _update_api_key(settings, 'elsevier_api_key', 'Elsevier API key', _check_elsevier_api_key, value=value)


def update_core_key(settings: dict[str, Any] | Literal[True] = True,
                    value: str | None = None) -> None:
    """Prompt for and save a CORE API key.

    Parameters
    ----------
    settings : dict[str, Any] or Literal[True], default=True
        Mapping to update in place, or ``True`` to load the current settings.
    value : str or None, optional
        Value to save without prompting; ``None`` prompts on stdin.
    """
    _update_api_key(settings, 'core_api_key', 'CORE API key', value=value)


def update_core_rate(settings: dict[str, Any] | Literal[True] = True,
                     value: str | None = None) -> None:
    """Prompt for and save the seconds to leave between CORE requests.

    CORE grants registered organisations and members a faster pace than the
    free allowance, agreed individually and published nowhere, so the pace is
    entered directly rather than derived from a membership level.

    Parameters
    ----------
    settings : dict[str, Any] or Literal[True], default=True
        Mapping to update in place, or ``True`` to load the current settings.
    value : str or None, optional
        Seconds to save without prompting; blank restores the free allowance.
        ``None`` prompts on stdin.

    Raises
    ------
    ValueError
        If the entered value is not a positive number.
    """
    from paperminertoolkit.providers.core import (CORE_FREE_SINGLE_PER_WINDOW,
                                                  CORE_MIN_INTERVAL,
                                                  CORE_WINDOW_SECONDS)
    if settings is True:
        settings = _saved_settings()
    if value is None:
        _show_current_setting(settings, 'core_min_interval',
                              'CORE seconds between requests', secret=False)
        print(f'Leave blank for CORE\'s free allowance of {CORE_FREE_SINGLE_PER_WINDOW} requests '
              f'per {CORE_WINDOW_SECONDS:.0f} seconds, which is {CORE_MIN_INTERVAL:.1f} seconds '
              'between requests. Enter a smaller number only if CORE granted you a faster rate.')
        value = input('Enter seconds between CORE requests: ')
    entered = value.strip()
    if not entered:
        settings.pop('core_min_interval', None)
        _save_settings(settings)
        return
    try:
        interval = float(entered)
    except ValueError as error:
        raise ValueError('CORE request pace must be a number of seconds.') from error
    if interval <= 0:
        raise ValueError('CORE request pace must be greater than zero seconds.')
    settings['core_min_interval'] = interval
    _save_settings(settings)


def _update_email(settings: dict[str, Any] | Literal[True], key: str, label: str,
                  value: str | None = None) -> None:
    """Prompt for and persist one provider contact email.

    Parameters
    ----------
    settings : dict[str, Any] or Literal[True]
        Mapping to update in place, or ``True`` to load the current settings.
    key : str
        Settings key containing the contact address.
    label : str
        Provider label used in the display, prompt, and validation error.
    value : str or None, optional
        Address to save without prompting; ``None`` prompts on stdin.

    Raises
    ------
    ValueError
        If the entered address does not contain an ``@`` character.
    """
    if settings is True:
        settings = _saved_settings()
    if value is None:
        _show_current_setting(settings, key, label, secret=False)
        value = input(f'Enter {label}: ')
    email = value.strip()
    if '@' not in email:
        raise ValueError(f'{label} must be a valid email address.')
    settings[key] = email
    _save_settings(settings)


def update_unpaywall_email(settings: dict[str, Any] | Literal[True] = True,
                           value: str | None = None) -> None:
    """Prompt for and save the email address sent to Unpaywall.

    Parameters
    ----------
    settings : dict[str, Any] or Literal[True], default=True
        Mapping to update in place, or ``True`` to load the current settings.
    value : str or None, optional
        Value to save without prompting; ``None`` prompts on stdin.

    Raises
    ------
    ValueError
        If the entered value does not contain an ``@`` character.
    """
    _update_email(settings, 'unpaywall_email', 'Unpaywall email', value=value)


def update_crossref_email(settings: dict[str, Any] | Literal[True] = True,
                          value: str | None = None) -> None:
    """Prompt for and save the contact email sent to Crossref.

    Crossref asks that automated clients identify themselves with a contact
    address. The same address is reused as the OpenAlex ``mailto`` parameter.

    Parameters
    ----------
    settings : dict[str, Any] or Literal[True], default=True
        Mapping to update in place, or ``True`` to load the current settings.
    value : str or None, optional
        Value to save without prompting; ``None`` prompts on stdin.

    Raises
    ------
    ValueError
        If the entered value does not contain an ``@`` character.
    """
    _update_email(settings, 'crossref_email', 'Crossref email', value=value)


def update_ncbi_key(settings: dict[str, Any] | Literal[True] = True,
                    value: str | None = None) -> None:
    """Prompt for and save an NCBI E-utilities API key.

    The key is optional. PubMed and PMC serve unauthenticated clients at three
    requests per second; a key raises that ceiling to ten.

    Parameters
    ----------
    settings : dict[str, Any] or Literal[True], default=True
        Mapping to update in place, or ``True`` to load the current settings.
    value : str or None, optional
        Value to save without prompting; ``None`` prompts on stdin.
    """
    _update_api_key(settings, 'ncbi_api_key', 'NCBI API key', value=value)


def update_ncbi_email(settings: dict[str, Any] | Literal[True] = True,
                      value: str | None = None) -> None:
    """Prompt for and save the contact email sent to NCBI E-utilities.

    NCBI asks that automated clients identify themselves so it can warn a
    contact before blocking an address. PaperMinerToolkit falls back to the Crossref
    address when no NCBI-specific address is configured.

    Parameters
    ----------
    settings : dict[str, Any] or Literal[True], default=True
        Mapping to update in place, or ``True`` to load the current settings.
    value : str or None, optional
        Value to save without prompting; ``None`` prompts on stdin.

    Raises
    ------
    ValueError
        If the entered value does not contain an ``@`` character.
    """
    _update_email(settings, 'ncbi_email', 'NCBI email', value=value)


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
    from paperminertoolkit.providers import openalex

    try:
        response = requests.get(openalex.RATE_LIMIT_URL,
                                params={'api_key': api_key},
                                headers=openalex.request_headers(),
                                timeout=30)
    except requests.RequestException:
        return True
    return response.status_code != 401


def update_openalex_key(settings: dict[str, Any] | Literal[True] = True,
                        value: str | None = None) -> None:
    """Prompt for, validate, and save an OpenAlex API key.

    OpenAlex permits unauthenticated requests with a smaller daily credit
    budget; a free API key increases that budget.

    Parameters
    ----------
    settings : dict[str, Any] or Literal[True], default=True
        Mapping to update in place, or ``True`` to load the current settings.
    value : str or None, optional
        Value to save without prompting; ``None`` prompts on stdin.

    Raises
    ------
    ValueError
        If OpenAlex explicitly rejects the entered key.
    """
    _update_api_key(
        settings, 'openalex_api_key', 'OpenAlex API key', _check_openalex_api_key,
        strip=True, value=value,
    )
