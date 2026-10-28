"""Shared connection conventions for Anthropic Messages requests."""

from __future__ import annotations

ANTHROPIC_VERSION = '2023-06-01'
DEFAULT_ANTHROPIC_BASE_URL = 'https://api.anthropic.com'


def request_headers(api_key: str) -> dict[str, str]:
    """Build headers for an authenticated Anthropic request.

    Parameters
    ----------
    api_key : str
        API key already validated by the caller.

    Returns
    -------
    dict[str, str]
        Authentication, API version, and JSON content headers.
    """
    return {
        'x-api-key': api_key,
        'anthropic-version': ANTHROPIC_VERSION,
        'content-type': 'application/json',
    }


def messages_url(base_url: str | None) -> str:
    """Resolve the Messages endpoint from an API root or versioned base.

    Parameters
    ----------
    base_url : str or None
        API root or base ending in ``/v1``. Empty values use Anthropic's
        default API root. A full Messages endpoint is not a supported base.

    Returns
    -------
    str
        Messages endpoint with trailing base slashes removed.
    """
    base_url = (base_url or DEFAULT_ANTHROPIC_BASE_URL).rstrip('/')
    return f'{base_url}/messages' if base_url.endswith('/v1') else f'{base_url}/v1/messages'
