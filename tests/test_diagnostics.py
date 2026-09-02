"""Tests for the provider configuration and reachability report."""

from __future__ import annotations

from typing import NoReturn

import pytest

import paperminertoolkit.workflows.diagnostics as diagnostics
from paperminertoolkit.providers import registry


def test_every_free_provider_declares_a_probe() -> None:
    """Give each source a probe unless asking it would cost something.

    A source with no probe is reported as unchecked rather than as working, so
    a missing one silently downgrades the report. Only a source whose sole
    route is metered is allowed to have none.
    """
    unprobed = {name for name in registry.SOURCES if not registry.SOURCES[name].probe}
    assert unprobed == {'openalex-content'}
    for name in registry.SOURCES:
        probe = registry.resolve_probe(name)
        assert probe is None or callable(probe)


def test_a_missing_optional_credential_does_not_stop_the_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ask a provider that answers without a credential whether it answers.

    Whether a provider responds is the question this report exists for, so a
    provider that works unconfigured must still be probed. Reporting it as
    merely "not set up" would hide the answer, and several providers here need
    no credential at all or only go faster with one.
    """
    monkeypatch.setattr(diagnostics, 'load_settings', lambda: {})
    monkeypatch.setattr(diagnostics.os, 'environ', {})
    monkeypatch.setattr(diagnostics.registry, 'resolve_probe',
                        lambda name: lambda: 'answered')

    rows = {row.name: row for row in diagnostics.provider_status(['pubmed', 'elsevier', 'arxiv'])}

    # PubMed's key is optional, so it is asked, and told what the key adds.
    assert rows['pubmed'].state == diagnostics.OK
    assert 'run pmt config ncbi-key for more' in rows['pubmed'].detail
    assert rows['pubmed'].is_problem is False
    # arXiv needs no credential at all.
    assert rows['arxiv'].state == diagnostics.OK
    assert rows['arxiv'].credential == ''
    # Elsevier's key is required, so there is nothing to ask.
    assert rows['elsevier'].state == diagnostics.NOT_SET_UP
    assert 'unusable without it' in rows['elsevier'].detail
    assert rows['elsevier'].is_problem is False


def test_a_configured_provider_that_fails_is_the_only_reported_fault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Separate a service that is down from a credential that is absent."""
    def refuse() -> NoReturn:
        """Fail the way a bot challenge does."""
        raise RuntimeError('chemRxiv refused the request with 403\nsecond line dropped')

    monkeypatch.setattr(diagnostics, 'load_settings', lambda: {})
    monkeypatch.setattr(diagnostics.os, 'environ', {})
    monkeypatch.setattr(diagnostics.registry, 'resolve_probe', lambda name: refuse)

    rows = {row.name: row for row in diagnostics.provider_status(['chemrxiv', 'core'])}

    assert rows['chemrxiv'].state == diagnostics.NOT_RESPONDING
    assert rows['chemrxiv'].is_problem is True
    # Only the first line is kept, so one row stays one row.
    assert rows['chemrxiv'].detail == 'chemRxiv refused the request with 403'
    # A required credential that is absent is a choice, not a fault.
    assert rows['core'].is_problem is False


def test_a_long_failure_is_truncated_and_an_empty_one_is_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep one failure to one line, however the provider phrased it."""
    monkeypatch.setattr(diagnostics, 'load_settings', lambda: {'ncbi_api_key': 'k'})
    monkeypatch.setattr(diagnostics.os, 'environ', {})

    def verbose() -> NoReturn:
        """Fail at length."""
        raise RuntimeError('x' * 400)

    monkeypatch.setattr(diagnostics.registry, 'resolve_probe', lambda name: verbose)
    row = diagnostics.provider_status(['pubmed'])[0]
    assert len(row.detail) == 120
    assert row.detail.endswith('...')

    def silent() -> NoReturn:
        """Fail with nothing to say."""
        raise TimeoutError('')

    monkeypatch.setattr(diagnostics.registry, 'resolve_probe', lambda name: silent)
    assert diagnostics.provider_status(['pubmed'])[0].detail == 'TimeoutError'


def test_no_probe_reports_configuration_without_requesting_anything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Answer offline, so a check can run where the network cannot."""
    def explode() -> NoReturn:
        """Fail if a probe is made at all."""
        raise AssertionError('probe should not run')

    monkeypatch.setattr(diagnostics, 'load_settings', lambda: {'ncbi_api_key': 'k'})
    monkeypatch.setattr(diagnostics.os, 'environ', {})
    monkeypatch.setattr(diagnostics.registry, 'resolve_probe', lambda name: explode)

    rows = {row.name: row for row in
            diagnostics.provider_status(['pubmed', 'arxiv'], probe=False)}

    assert {row.state for row in rows.values()} == {diagnostics.NOT_PROBED}
    assert rows['pubmed'].detail == 'configured'
    assert rows['arxiv'].detail == 'no credential needed'
    assert all(row.seconds == 0.0 for row in rows.values())


def test_a_metered_only_source_is_reported_rather_than_charged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never spend a metered request to prove a metered request can be spent."""
    monkeypatch.setattr(diagnostics, 'load_settings', lambda: {'openalex_api_key': 'k'})
    monkeypatch.setattr(diagnostics.os, 'environ', {})

    row = diagnostics.provider_status(['openalex-content'])[0]

    assert row.state == diagnostics.NOT_PROBED
    assert row.detail == 'not probed, as its only route is metered'
    assert row.is_problem is False


def test_probes_report_what_each_provider_reveals(monkeypatch: pytest.MonkeyPatch) -> None:
    """Say more than "answered" where the response carries something useful."""
    monkeypatch.setattr(diagnostics.crossref, 'resolve_email', lambda *_, **__: '')
    monkeypatch.setattr(diagnostics.crossref, 'work_by_doi', lambda *_, **__: {})
    assert diagnostics._probe_crossref() == 'public pool, 5/s'
    monkeypatch.setattr(diagnostics.crossref, 'resolve_email', lambda *_, **__: 'a@b.com')
    assert diagnostics._probe_crossref() == 'polite pool, 10/s'

    monkeypatch.setattr(diagnostics.pubmed, 'esearch', lambda *_, **__: ([], ''))
    monkeypatch.setattr(diagnostics.pubmed, 'configured_api_key', lambda *_, **__: None)
    assert diagnostics._probe_pubmed() == '3/s unauthenticated'
    monkeypatch.setattr(diagnostics.pubmed, 'configured_api_key', lambda *_, **__: 'k')
    assert diagnostics._probe_pubmed() == '10/s with a key'

    monkeypatch.setattr(diagnostics.openalex, 'configured_api_key', lambda *_, **__: 'k')
    monkeypatch.setattr(diagnostics.openalex, 'get_work', lambda *_, **__: {})
    monkeypatch.setattr(diagnostics.openalex, 'budget_status', lambda: None)
    assert diagnostics._probe_openalex() == 'answered'
    monkeypatch.setattr(diagnostics.openalex, 'budget_status',
                        lambda: diagnostics.openalex.provider.Budget(remaining=9900, limit=10000))
    assert diagnostics._probe_openalex() == '9,900 of 10,000 daily credits left'

    monkeypatch.setattr(diagnostics.elsevier, 'configured_api_key', lambda *_, **__: 'k')
    monkeypatch.setattr(diagnostics.elsevier, 'check_api_key', lambda *_, **__: True)
    monkeypatch.setattr(diagnostics.elsevier, 'quota_status', lambda *_, **__: None)
    assert diagnostics._probe_elsevier() == 'key accepted'
    monkeypatch.setattr(diagnostics.elsevier, 'check_api_key', lambda *_, **__: False)
    with pytest.raises(RuntimeError, match='rejected the API key'):
        diagnostics._probe_elsevier()

    for probe, module, function in [
        (diagnostics._probe_core, diagnostics.core, 'search_page'),
        (diagnostics._probe_unpaywall, diagnostics.unpaywall, 'get_work'),
        (diagnostics._probe_arxiv, diagnostics.arxiv, 'request_xml'),
        (diagnostics._probe_medrxiv, diagnostics.medrxiv, 'request_json'),
        (diagnostics._probe_biorxiv, diagnostics.biorxiv, 'request_json'),
        (diagnostics._probe_chemrxiv, diagnostics.chemrxiv, 'request_json'),
    ]:
        monkeypatch.setattr(module, function, lambda *_, **__: {})
        assert probe() == 'answered'


def test_elsevier_probe_reports_the_quota_the_response_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Say what is left of the weekly allowance, not merely that the key works.

    Elsevier's quota is the allowance that runs out mid-run, so a check that
    can report it should, rather than making someone find out by failing.
    """
    from paperminertoolkit.providers import base as provider

    monkeypatch.setattr(diagnostics.elsevier, 'configured_api_key', lambda *_, **__: 'k')
    monkeypatch.setattr(diagnostics.elsevier, 'check_api_key', lambda *_, **__: True)
    monkeypatch.setattr(diagnostics.elsevier, 'quota_status',
                        lambda *_, **__: provider.Budget(remaining=19_998, limit=20_000))

    assert diagnostics._probe_elsevier() == (
        'key accepted, 19,998 of 20,000 requests left this period')


def test_a_malformed_probe_target_is_rejected() -> None:
    """Fail loudly on a probe target that cannot be resolved."""
    from dataclasses import replace

    with pytest.raises(ValueError, match='invalid probe for arxiv'):
        _resolve(replace(registry.SOURCES['arxiv'], probe='not-a-target'))


def _resolve(entry: registry.Source) -> object:
    """Resolve one entry's probe through the registry, for the test above."""
    original = registry.SOURCES[entry.name]
    registry.SOURCES[entry.name] = entry
    try:
        return registry.resolve_probe(entry.name)
    finally:
        registry.SOURCES[entry.name] = original


def test_a_probe_target_that_is_not_callable_is_rejected() -> None:
    """Refuse a target that resolves to something that cannot be called."""
    from dataclasses import replace

    entry = replace(registry.SOURCES['arxiv'],
                    probe='paperminertoolkit.workflows.diagnostics:PROBE_DOI')
    with pytest.raises(TypeError, match='is not callable'):
        _resolve(entry)
