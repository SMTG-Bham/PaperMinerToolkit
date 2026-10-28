"""Report which providers are configured and which are actually answering.

A run that fails partway through rarely says whether a credential is missing, a
credential is wrong, or a service is simply down. Those need different fixes, so
this module reports them as three different states rather than collapsing them
into "not working": a provider is `ok`, `not set up`, or `not responding`.

"Not set up" is not always a problem, which is why it is not an error. Several
providers answer perfectly well without a credential and only go slower or get a
smaller budget for the want of one, so each row also says what the missing
credential would buy.

Each probe is the cheapest read-only request its provider documents, made
through that provider's own limiter so a check obeys the same pacing as a run.
A provider whose only route is metered declares no probe rather than spending a
request to prove it can spend a request.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from paperminertoolkit.providers import (arxiv, biorxiv, chemrxiv, core, crossref, elsevier,
                                         medrxiv, openalex, pubmed, registry, unpaywall)
from paperminertoolkit.settings import load_settings

OK = 'ok'
NOT_SET_UP = 'not set up'
NOT_RESPONDING = 'not responding'
NOT_PROBED = 'not probed'
# A DOI every metadata provider knows, used so a probe exercises a real lookup
# rather than an endpoint that answers whatever it is asked.
PROBE_DOI = '10.1038/nature12373'
# A CORE record used only to prove CORE answers. Its content does not matter,
# and a withdrawn record still answers, so this needs no maintenance.
CORE_PROBE_ID = '24003915'


@dataclass(frozen=True, slots=True)
class ProviderStatus:
    """What one provider's configuration and last probe reported.

    Parameters
    ----------
    name : str
        Registry source name.
    label : str
        Display name.
    credential : str
        Human-readable name of the credential this provider uses, or an empty
        string when it needs none.
    state : str
        One of :data:`OK`, :data:`NOT_SET_UP`, :data:`NOT_RESPONDING`, or
        :data:`NOT_PROBED`.
    detail : str
        What the state means for this provider, and what to do about it.
    seconds : float, default=0.0
        Time the probe took, when one ran.
    """

    name: str
    label: str
    credential: str
    state: str
    detail: str
    seconds: float = 0.0

    @property
    def is_problem(self) -> bool:
        """Return whether this row is a fault rather than a choice.

        A provider that is configured and not answering is a fault. One that is
        simply not set up is not: several work without a credential, and a run
        may legitimately not use the rest.
        """
        return self.state == NOT_RESPONDING


def _probe_crossref() -> str:
    """Look up one work, reporting which pool the request qualified for.

    Crossref reports an absent contact address once per process. Here that
    would print above the table while the row is about to say the same thing
    more precisely, so the notice is marked as already given.
    """
    crossref._reported_missing_contact_address = True
    email = crossref.resolve_email()
    crossref.work_by_doi(PROBE_DOI, email=email)
    return 'polite pool, 10/s' if email else 'public pool, 5/s'


def _probe_openalex() -> str:
    """Look up one work, reporting the daily credit budget it reveals."""
    key = openalex.configured_api_key()
    openalex.get_work(f'doi:{PROBE_DOI}', api_key=key)
    budget = openalex.budget_status()
    if budget is None:
        return 'answered'
    return f'{budget.remaining:,} of {budget.limit:,} daily credits left'


def _probe_pubmed() -> str:
    """Search for one record, reporting the rate the credential earns."""
    key = pubmed.configured_api_key()
    pubmed.esearch('graphene', retmax=1, api_key=key, email=pubmed.configured_email())
    return '10/s with a key' if key else '3/s unauthenticated'


def _probe_elsevier() -> str:
    """Validate the key, reporting any quota the response declared."""
    key = elsevier.configured_api_key()
    if not elsevier.check_api_key(key):
        raise RuntimeError('Elsevier rejected the API key')
    quota = elsevier.quota_status('https://api.elsevier.com/content/search/scopus', key)
    if quota is None:
        return 'key accepted'
    return f'key accepted, {quota.remaining:,} of {quota.limit:,} requests left this period'


def _probe_core() -> str:
    """Look one record up by identifier, which CORE answers quickly.

    CORE's search endpoint takes 45 to 70 seconds for a broad query, which is
    the service rather than the pacing, and far too slow for a status check. A
    single-record lookup answers in a few seconds. A record that has since been
    withdrawn still proves CORE responded, so an empty result is a success
    here; only a failed request is not.
    """
    core.get_work(CORE_PROBE_ID, api_key=core.configured_api_key())
    return 'answered'


def _probe_openalex_content() -> str:
    """Check the cached-PDF route, which is the one probe that costs.

    Content routes have no free allowance, and a HEAD is billed exactly as a
    GET is, so this spends 100 credits of the daily budget. It saves the
    transfer, not the charge, which is why the cost is reported rather than
    hidden.
    """
    key = openalex.configured_api_key() or ''
    work = openalex.get_work(f'doi:{PROBE_DOI}', api_key=key)
    url = openalex.cached_pdf_url(work or {})
    if not url:
        raise RuntimeError('OpenAlex holds no cached PDF for the probe record')
    size = openalex.cached_pdf_available(url, key)
    return f'{size:,} byte PDF available; cost 100 credits'


def _probe_unpaywall() -> str:
    """Look up one DOI through Unpaywall."""
    unpaywall.get_work(PROBE_DOI)
    return 'answered'


def _probe_arxiv() -> str:
    """Request one search result from arXiv."""
    arxiv.request_xml(arxiv.BASE_URL,
                      params={'search_query': 'all:graphene', 'start': 0, 'max_results': 1})
    return 'answered'


def _probe_medrxiv() -> str:
    """Request one posting record from medRxiv."""
    medrxiv.request_json(medrxiv.details_url('10.1101/2024.05.31.24307874'))
    return 'answered'


def _probe_biorxiv() -> str:
    """Request one posting record from bioRxiv."""
    biorxiv.request_json(biorxiv.details_url('10.1101/2023.03.30.534894'))
    return 'answered'


def _probe_chemrxiv() -> str:
    """Request the category list, which is chemRxiv's cheapest endpoint."""
    chemrxiv.request_json(chemrxiv.categories_url())
    return 'answered'


def _credential_state(entry: registry.Source, settings: Mapping[str, str]) -> tuple[bool, str]:
    """Report whether a provider's credential is present, and what it is worth.

    Parameters
    ----------
    entry : registry.Source
        Registry entry to inspect.
    settings : Mapping[str, str]
        Loaded settings.

    Returns
    -------
    tuple[bool, str]
        Whether a credential is configured, and how the credential is named for
        display. A provider needing none reports ``True`` and an empty name.
    """
    if not entry.credential:
        return True, ''
    configured = bool(settings.get(entry.credential) or os.environ.get(entry.credential_env))
    return configured, entry.credential_env or entry.credential


def _missing_credential_note(entry: registry.Source) -> str:
    """Say what a missing credential costs, which is not always the provider.

    Parameters
    ----------
    entry : registry.Source
        Registry entry whose credential is absent.

    Returns
    -------
    str
        What to run to supply it.
    """
    return f'run {entry.setup_command or f"set {entry.credential_env}"} for more'


def provider_status(names: Sequence[str] | None = None, probe: bool = True) -> list[ProviderStatus]:
    """Report the configuration and reachability of every known provider.

    Parameters
    ----------
    names : Sequence[str] or None, optional
        Providers to report on. ``None`` reports every registered source.
    probe : bool, default=True
        Whether to make one read-only request per configured provider.

    Returns
    -------
    list[ProviderStatus]
        One row per provider, in registry order.
    """
    settings = load_settings()
    rows = []
    for name in names if names is not None else registry.SOURCES:
        entry = registry.SOURCES[name]
        configured, credential = _credential_state(entry, settings)
        # Only a required credential stops the check. A provider that answers
        # without one is still asked, because whether it answers is the
        # question, and reporting it as merely "not set up" would hide that.
        if not configured and entry.credential_required:
            command = entry.setup_command or f'set {entry.credential_env}'
            rows.append(ProviderStatus(name, entry.label, credential, NOT_SET_UP,
                                       f'unusable without it; run {command}'))
            continue
        if not probe:
            if not credential:
                detail = 'no credential needed'
            else:
                detail = 'configured' if configured else 'usable unconfigured'
            rows.append(ProviderStatus(name, entry.label, credential, NOT_PROBED, detail))
            continue
        checker = registry.resolve_probe(name)
        if checker is None:
            rows.append(ProviderStatus(name, entry.label, credential, NOT_PROBED,
                                       'no probe registered'))
            continue
        start = time.monotonic()
        try:
            detail = checker()
            state = OK
            if not configured:
                detail = f'{detail}; {_missing_credential_note(entry)}'
        except Exception as error:  # noqa: BLE001 - every failure is a report, not a raise
            # An exception may carry no message at all, which splitlines()
            # reports as no lines rather than one empty one.
            lines = str(error).strip().splitlines()
            # The whole reason is kept. A provider's refusal often carries the
            # remedy in its second half, so shortening it here would throw away
            # the useful part; the display shortens it instead, and reprints it
            # in full underneath.
            detail = (lines[0] if lines else '') or type(error).__name__
            state = NOT_RESPONDING
        rows.append(ProviderStatus(name, entry.label, credential, state, detail,
                                   time.monotonic() - start))
    return rows
