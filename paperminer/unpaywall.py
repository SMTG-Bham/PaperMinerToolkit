"""Request helpers for the Unpaywall API used by PaperMiner.

Unpaywall holds no papers of its own. It answers one question, for a DOI: where
is a legally free copy of this? That makes it the first PDF source a download
run tries, and not a search or enrichment source at all.

A contact address is required rather than a key: Unpaywall identifies a client
by an email in the query string and refuses a request without one. Store one
with ``ps_unpaywall_email`` or in ``UNPAYWALL_EMAIL``.

A record lists every open location it knows of and flags the one it considers
best. :func:`pdf_candidates` returns them best-first, because an individual
location may be a dead link, a landing page, or a copy the host has since taken
down, while another for the same paper still resolves.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, TypeAlias
from urllib.parse import quote

from paperminer import provider

BASE_URL = 'https://api.unpaywall.org/v2'
UNPAYWALL_MIN_INTERVAL = 0.1
_UnpaywallRecord: TypeAlias = dict[str, Any]
LIMITER = provider.RateLimiter(UNPAYWALL_MIN_INTERVAL)


def configured_email(settings: Mapping[str, str] | None = None) -> str:
    """Return the configured Unpaywall contact address.

    Parameters
    ----------
    settings : Mapping[str, str] or None, optional
        Loaded PaperMiner settings. Read from disk when omitted.

    Returns
    -------
    str
        Contact address, or an empty string when none is configured.
    """
    from paperminer.settings import load_settings
    settings = settings if settings is not None else load_settings()
    return str(settings.get('unpaywall_email') or os.environ.get('UNPAYWALL_EMAIL') or '')


def work_url(doi: object) -> str:
    """Build the Unpaywall lookup URL for one DOI.

    Parameters
    ----------
    doi : object
        DOI to look up.

    Returns
    -------
    str
        Lookup URL, or an empty string when no DOI is present.
    """
    identifier = str(doi or '').strip()
    return f'{BASE_URL}/{quote(identifier, safe="")}' if identifier else ''


def get_work(doi: object,
             email: str = '',
             session: provider.HTTPClient | None = None,
             timeout: float = provider.DEFAULT_TIMEOUT,
             attempts: int = provider.DEFAULT_ATTEMPTS) -> _UnpaywallRecord | None:
    """Fetch Unpaywall's record for one DOI.

    Parameters
    ----------
    doi : object
        DOI to look up.
    email : str, default=''
        Contact address to send. Read from settings when omitted.
    session : provider.HTTPClient or None, optional
        HTTP client exposing a ``get`` method.
    timeout : float, default=60.0
        Request timeout in seconds.
    attempts : int, default=4
        Maximum number of request attempts.

    Returns
    -------
    dict[str, Any] or None
        Unpaywall record, or ``None`` when it knows nothing of the DOI.

    Raises
    ------
    ValueError
        If no contact address is configured.
    RuntimeError
        If the request cannot be completed.
    """
    url = work_url(doi)
    if not url:
        return None
    address = email or configured_email()
    if not address:
        raise ValueError('Unpaywall email is not configured. Run ps_unpaywall_email first.')
    return provider.request_mapping(url, label='Unpaywall', limiter=LIMITER,
                                    params={'email': address}, session=session,
                                    timeout=timeout, attempts=attempts)


def pdf_candidates(work: Mapping[str, Any]) -> list[str]:
    """Return candidate PDF URLs for an Unpaywall record, best first.

    Every known location is offered rather than only the best one, because an
    individual host may have moved or withdrawn its copy while another still
    serves the same paper.

    Parameters
    ----------
    work : Mapping[str, Any]
        Unpaywall record.

    Returns
    -------
    list[str]
        Deduplicated PDF URLs, most authoritative first.
    """
    candidates = [(work.get('best_oa_location') or {}).get('url_for_pdf')]
    for location in work.get('oa_locations') or []:
        candidates.append((location or {}).get('url_for_pdf'))
    return list(dict.fromkeys(url for url in candidates if url))


def is_oa(work: Mapping[str, Any]) -> bool:
    """Report whether Unpaywall considers a paper open access.

    Parameters
    ----------
    work : Mapping[str, Any]
        Unpaywall record.

    Returns
    -------
    bool
        Whether a free copy is known.
    """
    return bool(work.get('is_oa'))


def oa_status(work: Mapping[str, Any]) -> str:
    """Return Unpaywall's open-access colour for a paper.

    Parameters
    ----------
    work : Mapping[str, Any]
        Unpaywall record.

    Returns
    -------
    str
        Status such as ``gold`` or ``green``, or an empty string.
    """
    return provider.clean_text(work.get('oa_status'))
