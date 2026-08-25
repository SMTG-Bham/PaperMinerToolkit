"""Unit tests for Unpaywall API request helpers."""

from __future__ import annotations

from typing import Any

import pytest

import paperminer.providers.unpaywall as unpaywall
from paperminer.providers import base as provider

from tests.doubles import FakeResponse, FakeSession


def record(**overrides: Any) -> dict[str, Any]:
    """Return an Unpaywall record for the tests to read.

    Parameters
    ----------
    **overrides : Any
        Fields to replace on the record.

    Returns
    -------
    dict[str, Any]
        Unpaywall record.
    """
    payload = {
        'doi': '10.1234/example',
        'is_oa': True,
        'oa_status': 'gold',
        'best_oa_location': {'url_for_pdf': 'https://publisher.example/best.pdf'},
        'oa_locations': [
            {'url_for_pdf': 'https://publisher.example/best.pdf'},
            {'url_for_pdf': 'https://repository.example/copy.pdf'},
            {'url_for_pdf': None},
        ],
    }
    payload.update(overrides)
    return payload


def test_configured_email_prefers_settings_then_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read the stored address first, and fall back to the environment."""
    monkeypatch.delenv('UNPAYWALL_EMAIL', raising=False)
    assert unpaywall.configured_email({'unpaywall_email': 'me@example.com'}) == 'me@example.com'
    assert unpaywall.configured_email({}) == ''

    monkeypatch.setenv('UNPAYWALL_EMAIL', 'env@example.com')
    assert unpaywall.configured_email({}) == 'env@example.com'


def test_work_url_encodes_the_doi() -> None:
    """Keep a DOI's slash out of the path so the lookup addresses one record."""
    assert unpaywall.work_url('10.1234/example') == f'{unpaywall.BASE_URL}/10.1234%2Fexample'
    assert unpaywall.work_url(' 10.1/x ') == f'{unpaywall.BASE_URL}/10.1%2Fx'
    assert unpaywall.work_url('') == ''
    assert unpaywall.work_url(None) == ''


def test_get_work_sends_the_contact_address_unpaywall_requires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Identify the client, because Unpaywall refuses an anonymous request."""
    monkeypatch.setattr(unpaywall, 'configured_email', lambda *_: 'me@example.com')
    session = FakeSession([FakeResponse(payload=record())])

    assert unpaywall.get_work('10.1234/example', session=session)['oa_status'] == 'gold'
    assert session.calls[0]['params'] == {'email': 'me@example.com'}
    assert session.calls[0]['headers']['User-Agent'] == provider.USER_AGENT


def test_get_work_requires_a_contact_address(monkeypatch: pytest.MonkeyPatch) -> None:
    """Say what is missing rather than spending a request that will be refused."""
    monkeypatch.setattr(unpaywall, 'configured_email', lambda *_: '')
    session = FakeSession([])

    with pytest.raises(ValueError, match='Unpaywall email is not configured'):
        unpaywall.get_work('10.1234/example', session=session)
    assert session.calls == []


def test_get_work_skips_a_row_with_no_doi() -> None:
    """Ask nothing for a paper Unpaywall cannot be keyed on."""
    session = FakeSession([])
    assert unpaywall.get_work('', email='me@example.com', session=session) is None
    assert session.calls == []


def test_get_work_reads_an_unknown_doi_as_nothing_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read a 404 as an absent record rather than as a failure."""
    monkeypatch.setattr(unpaywall, 'configured_email', lambda *_: 'me@example.com')
    session = FakeSession([FakeResponse(status_code=404)])
    assert unpaywall.get_work('10.1234/missing', session=session) is None


def test_pdf_candidates_lead_with_the_best_location_and_deduplicate() -> None:
    """Offer every known copy, best first, because one host may have moved on."""
    assert unpaywall.pdf_candidates(record()) == [
        'https://publisher.example/best.pdf',
        'https://repository.example/copy.pdf',
    ]
    assert unpaywall.pdf_candidates({'oa_locations': [{'url_for_pdf': 'only.pdf'}]}) == ['only.pdf']
    assert unpaywall.pdf_candidates({}) == []
    assert unpaywall.pdf_candidates({'best_oa_location': None, 'oa_locations': None}) == []


def test_open_access_flags_read_the_record() -> None:
    """Report the record's own view of whether a free copy exists."""
    assert unpaywall.is_oa(record()) is True
    assert unpaywall.is_oa({'is_oa': False}) is False
    assert unpaywall.is_oa({}) is False
    assert unpaywall.oa_status(record()) == 'gold'
    assert unpaywall.oa_status({}) == ''
