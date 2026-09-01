"""Shared HTTP plumbing for the PaperMinerToolkit data-source clients.

Every data source PaperMinerToolkit reads from -- Crossref, OpenAlex, PubMed,
Elsevier, CORE, Unpaywall, arXiv, medRxiv, bioRxiv, and chemRxiv -- needs the
same four things: a courtesy delay between requests, a bounded retry with
exponential backoff that honours ``Retry-After``, a way to tell a missing
record apart from a rejected request, and an injection point so tests can
answer without a network. Those were written once per source and drifted, so
they live here instead and each source supplies only what is actually its own:
its endpoint, its pace, and the handful of status codes that mean something
particular to it.

Pacing state is deliberately per-source rather than global. arXiv asks for three
seconds between requests and PubMed allows ten a second with a key, so one
shared window would slow every source to the most cautious one. Each module
therefore owns a :class:`RateLimiter`, and :func:`request` paces against the one
it is handed.

The retry policy is the one the majority of the sources already used: a 404 is a
missing record and returns ``None``, any other client error is terminal because
retrying it only spends another request, and 429 and server errors are retried
because they are the ones that pass. Sources that read a status code
differently -- chemRxiv's bot challenge answers 403, OpenAlex's 429 means the
daily credit budget is gone rather than "slow down" -- pass a ``client_error``
hook that is consulted first.
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import requests

from paperminertoolkit._version import __version__

USER_AGENT = f'PaperMinerToolkit/{__version__}'
DEFAULT_TIMEOUT = 60.0
DEFAULT_ATTEMPTS = 4
MAX_RETRY_DELAY = 60.0
# Errors worth another attempt. Providers whose decode step can fail in a
# further way extend this rather than wrapping the call.
RETRY_ERRORS: tuple[type[Exception], ...] = (requests.RequestException, ValueError)
# Values a provider may send in place of an absent field. They are not data, so
# text helpers report them as empty rather than passing the placeholder on.
NOT_AVAILABLE = frozenset({'', 'na', 'n/a', 'none', 'null'})
# Every limiter ever built, so a test run can reopen all of their windows
# without having to know which source modules it happened to import.
_LIMITERS: list['RateLimiter'] = []


@dataclass(frozen=True, slots=True)
class FullTextDocument:
    """Hold derived text and its original structured article document.

    Parameters
    ----------
    text : str
        Plain text derived from the provider document.
    content : str, optional
        Unmodified structured article response. Empty when the provider only
        supplies plain text.
    document_format : str, optional
        Provider-neutral structured format name, such as ``"jats"``,
        ``"elsevier-xml"``, or ``"tei"``.
    source_url : str, optional
        URL from which the structured document was retrieved.
    source_identifier : str, optional
        Provider-native identifier for the retrieved document.
    mime_type : str, default='application/xml'
        Media type of ``content``.
    metadata : Mapping[str, object] or None, optional
        Additional provenance supplied by the provider.
    """

    text: str
    content: str = ''
    document_format: str = ''
    source_url: str = ''
    source_identifier: str = ''
    mime_type: str = 'application/xml'
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def has_structured_content(self) -> bool:
        """Return whether this result carries a structured source document."""
        return bool(self.content.strip() and self.document_format.strip())


class ResponseLike(Protocol):
    """HTTP response surface the source clients rely on."""

    status_code: int
    headers: Mapping[str, str]
    text: str

    def json(self) -> Any:
        """Decode the response body as JSON.

        Returns
        -------
        Any
            Decoded JSON payload.
        """
        ...

    def raise_for_status(self) -> None:
        """Raise when the response has an unsuccessful status.

        Returns
        -------
        None
            Nothing is returned for a successful status.
        """
        ...


class HTTPClient(Protocol):
    """HTTP client surface accepted for dependency injection."""

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, object],
        headers: Mapping[str, str],
        timeout: float,
    ) -> ResponseLike:
        """Issue an HTTP GET request.

        Parameters
        ----------
        url : str
            Endpoint to request.
        params : Mapping[str, object]
            Query parameters for the request.
        headers : Mapping[str, str]
            Request headers.
        timeout : float
            Request timeout in seconds.

        Returns
        -------
        ResponseLike
            The provider's response.
        """
        ...


class RateLimiter:
    """Pace requests to one provider by a minimum interval.

    A provider's courtesy delay is counted per client against one service, so
    the window has to be shared by every request to that service and by no
    other. One instance per source module gives exactly that: arXiv's three
    seconds do not hold up PubMed, and two arXiv calls from different parts of
    the package still queue behind each other.

    Parameters
    ----------
    min_interval : float
        Minimum seconds between consecutive requests. Zero disables pacing.
    """

    def __init__(self, min_interval: float) -> None:
        """Store the interval and open the window.

        Parameters
        ----------
        min_interval : float
            Minimum seconds between consecutive requests.

        Returns
        -------
        None
            The limiter is initialized in place.
        """
        self.min_interval = min_interval
        self._last_request_at = 0.0
        _LIMITERS.append(self)

    def wait(self, interval: float | None = None) -> None:
        """Sleep until the window allows another request.

        Parameters
        ----------
        interval : float or None, optional
            Override for this call, used by providers whose pace depends on
            whether a credential is configured. Defaults to ``min_interval``.

        Returns
        -------
        None
            The window is advanced in place.
        """
        pace = self.min_interval if interval is None else interval
        now = time.monotonic()
        delay = pace - (now - self._last_request_at)
        if delay > 0:
            time.sleep(delay)
            now = time.monotonic()
        self._last_request_at = now

    def reset(self) -> None:
        """Reopen the window so the next request is not paced.

        Returns
        -------
        None
            The window is cleared in place.
        """
        self._last_request_at = 0.0


def reset_limiters() -> None:
    """Reopen every limiter's window.

    Pacing state outlives a single call by design, which would otherwise let
    one test's requests delay the next test's. Tests call this between cases.

    Returns
    -------
    None
        Every limiter is reset in place.
    """
    for limiter in _LIMITERS:
        limiter.reset()


def user_agent(mailto: str = '') -> str:
    """Build the user agent, in polite-pool form when given a contact address.

    Crossref and OpenAlex route requests that name a contact address onto a
    faster pool, and the address travels in the user agent rather than as a
    header of its own.

    Parameters
    ----------
    mailto : str, default=''
        Contact email address to advertise, if any.

    Returns
    -------
    str
        User agent string identifying PaperMinerToolkit and its version.
    """
    return f'{USER_AGENT} (mailto:{mailto})' if mailto else USER_AGENT


def default_headers(mailto: str = '') -> dict[str, str]:
    """Build the request headers every source sends.

    Parameters
    ----------
    mailto : str, default=''
        Contact email address to advertise, if any.

    Returns
    -------
    dict[str, str]
        Headers identifying PaperMinerToolkit and its version.
    """
    return {'User-Agent': user_agent(mailto)}


def _retry_delay(error: Exception, attempt: int) -> float:
    """Choose how long to wait before retrying a failed request.

    A provider that sends ``Retry-After`` is telling the client when it will
    answer again, so that is preferred over the backoff curve; it is clamped
    because a header asking for an hour would read as a hang.

    Parameters
    ----------
    error : Exception
        Error raised by the failed attempt.
    attempt : int
        Zero-based index of the attempt that failed.

    Returns
    -------
    float
        Seconds to wait before the next attempt.
    """
    headers = getattr(getattr(error, 'response', None), 'headers', {})
    retry_after = headers.get('Retry-After') if hasattr(headers, 'get') else None
    if not retry_after:
        return float(2 ** attempt)
    try:
        return min(max(float(retry_after), 0.0), MAX_RETRY_DELAY)
    except (TypeError, ValueError):
        return float(2 ** attempt)


def request(
    url: str,
    *,
    label: str,
    limiter: RateLimiter,
    params: Mapping[str, object] | None = None,
    headers: Mapping[str, str] | None = None,
    session: HTTPClient | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    attempts: int = DEFAULT_ATTEMPTS,
    interval: float | None = None,
    client_error: Callable[[ResponseLike], str] | None = None,
    missing_ok: bool = True,
    error_types: tuple[type[Exception], ...] = RETRY_ERRORS,
    on_response: Callable[[ResponseLike], None] | None = None,
) -> ResponseLike | None:
    """Request a provider endpoint with courtesy pacing and bounded retries.

    Parameters
    ----------
    url : str
        Endpoint URL to request.
    label : str
        Provider name as it should read in error messages, such as ``arXiv``.
    limiter : RateLimiter
        Pacing window for this provider.
    params : Mapping[str, object] or None, optional
        Query parameters for the request.
    headers : Mapping[str, str] or None, optional
        Request headers. Defaults to :func:`default_headers`.
    session : HTTPClient or None, optional
        HTTP client exposing a ``get`` method. Defaults to :mod:`requests`.
    timeout : float, default=60.0
        Request timeout in seconds.
    attempts : int, default=4
        Maximum number of request attempts.
    interval : float or None, optional
        Pacing override for this call, passed to :meth:`RateLimiter.wait`.
    client_error : callable or None, optional
        Consulted before the default client-error rule with the response, and
        returning the message to fail with when the provider reads that status
        as terminal, or an empty string to fall through.
    missing_ok : bool, default=True
        Whether a 404 means "no such record" and returns ``None``. Set false
        for a provider whose caller reads a rejected request from the failure
        instead.
    error_types : tuple of type, default=RETRY_ERRORS
        Exception types that count as a failed attempt rather than propagating.
    on_response : callable or None, optional
        Called with every response received, whatever its status and once per
        attempt, before the status is interpreted. This is where a provider
        reads what its response headers say about the state of the account,
        such as a remaining quota, which is reported on a refusal as much as on
        a success.

    Returns
    -------
    ResponseLike or None
        Successful response, or ``None`` for a 404 response.

    Raises
    ------
    RuntimeError
        If the request is rejected, or all request attempts fail.
    """
    client = session or requests
    sent = dict(headers) if headers is not None else default_headers()
    merged = dict(params or {})
    last_error: Exception | None = None
    for attempt in range(attempts):
        limiter.wait(interval)
        try:
            response = client.get(url, params=merged, headers=sent, timeout=timeout)
            if on_response is not None:
                on_response(response)
            if missing_ok and response.status_code == 404:
                return None
            terminal = client_error(response) if client_error is not None else ''
            if terminal:
                raise RuntimeError(terminal)
            if 400 <= response.status_code < 500 and response.status_code != 429:
                raise RuntimeError(f'{label} rejected the request with '
                                   f'{response.status_code} from {url}')
            response.raise_for_status()
            return response
        except error_types as error:
            last_error = error
            status = getattr(getattr(error, 'response', None), 'status_code', None)
            if status is not None and 400 <= status < 500 and status != 429:
                break
            if attempt + 1 == attempts:
                break
            time.sleep(_retry_delay(error, attempt))
    raise RuntimeError(f'{label} request failed after {attempts} attempts: '
                       f'{last_error}') from last_error


def request_payload(
    url: str,
    *,
    label: str,
    limiter: RateLimiter,
    decode_error: Callable[[ResponseLike, ValueError], str] | None = None,
    **kwargs: Any,
) -> Any:
    """Request a provider endpoint and decode its JSON body.

    Parameters
    ----------
    url : str
        Endpoint URL to request.
    label : str
        Provider name as it should read in error messages.
    limiter : RateLimiter
        Pacing window for this provider.
    decode_error : callable or None, optional
        Consulted with the response and the decoding error when the body is not
        JSON, and returning a more specific message, or an empty string to fall
        through to the generic one.
    **kwargs : Any
        Further keyword arguments for :func:`request`.

    Returns
    -------
    Any
        Decoded payload, or ``None`` for a 404 response.

    Raises
    ------
    RuntimeError
        If the request fails or the body is not well-formed JSON.
    """
    response = request(url, label=label, limiter=limiter, **kwargs)
    if response is None:
        return None
    try:
        return response.json()
    except ValueError as error:
        specific = decode_error(response, error) if decode_error is not None else ''
        if specific:
            raise RuntimeError(specific) from error
        raise RuntimeError(f'{label} returned malformed JSON: {error}') from error


def request_mapping(
    url: str,
    *,
    label: str,
    limiter: RateLimiter,
    **kwargs: Any,
) -> dict[str, Any] | None:
    """Request a provider endpoint that answers with a JSON object.

    Parameters
    ----------
    url : str
        Endpoint URL to request.
    label : str
        Provider name as it should read in error messages.
    limiter : RateLimiter
        Pacing window for this provider.
    **kwargs : Any
        Further keyword arguments for :func:`request_payload`.

    Returns
    -------
    dict[str, Any] or None
        Decoded payload, or ``None`` for a 404 response.

    Raises
    ------
    RuntimeError
        If the request fails, the body is not well-formed JSON, or the payload
        is not a JSON object.
    """
    payload = request_payload(url, label=label, limiter=limiter, **kwargs)
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise RuntimeError(f'{label} returned an unexpected payload of type '
                           f'{type(payload).__name__}')
    return dict(payload)


def request_xml(
    url: str,
    *,
    label: str,
    limiter: RateLimiter,
    **kwargs: Any,
) -> ET.Element | None:
    """Request a provider endpoint and parse its XML body.

    Parameters
    ----------
    url : str
        Endpoint URL to request.
    label : str
        Provider name as it should read in error messages.
    limiter : RateLimiter
        Pacing window for this provider.
    **kwargs : Any
        Further keyword arguments for :func:`request`.

    Returns
    -------
    xml.etree.ElementTree.Element or None
        Parsed document root, or ``None`` for a 404 or empty response.

    Raises
    ------
    RuntimeError
        If the request fails or the payload is not well-formed XML.
    """
    response = request(url, label=label, limiter=limiter, **kwargs)
    if response is None:
        return None
    text = response.text or ''
    if not text.strip():
        return None
    try:
        return ET.fromstring(text)
    except ET.ParseError as error:
        raise RuntimeError(f'{label} returned malformed XML: {error}') from error


def chunked(values: Sequence[str], size: int) -> Iterator[list[str]]:
    """Split a sequence into batches of at most ``size`` values.

    Parameters
    ----------
    values : Sequence[str]
        Values to split.
    size : int
        Maximum batch length. A size below one is read as one, so a miscomputed
        batch size cannot silently yield empty batches or loop forever.

    Yields
    ------
    list[str]
        Successive batches in the original order.
    """
    step = max(int(size), 1)
    for start in range(0, len(values), step):
        yield list(values[start:start + step])


def clean_text(value: object) -> str:
    """Normalize a provider text field, treating placeholders as absent.

    Parameters
    ----------
    value : object
        Raw provider value.

    Returns
    -------
    str
        Whitespace-collapsed text, empty when the value is missing or is one of
        the provider's not-available placeholders.
    """
    if value is None:
        return ''
    text = ' '.join(str(value).split())
    return '' if text.lower() in NOT_AVAILABLE else text
