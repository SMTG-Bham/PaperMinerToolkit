"""Tests for the provider configuration and reachability report."""

from __future__ import annotations

import inspect
from typing import NoReturn

import pytest

import paperminertoolkit.workflows.diagnostics as diagnostics
from paperminertoolkit.providers import registry


def test_every_provider_declares_a_probe() -> None:
    """Give every source a probe, so none is silently reported as unchecked."""
    assert not [name for name in registry.SOURCES if not registry.SOURCES[name].probe]
    for name in registry.SOURCES:
        assert callable(registry.resolve_probe(name))


def test_only_the_content_probe_costs_anything() -> None:
    """Keep the one billed probe identifiable, since it spends a real budget.

    Every other probe is free: OpenAlex's singleton lookup is documented as
    costing nothing, and the rest are unmetered services. Content routes have
    no free allowance at all, so that probe is the one a caller may want to
    avoid, which is what ``--no-probe`` is for.
    """
    assert diagnostics._probe_openalex_content.__doc__ is not None
    assert 'costs' in diagnostics._probe_openalex_content.__doc__


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
    assert 'second line' not in rows['chemrxiv'].detail
    # A required credential that is absent is a choice, not a fault.
    assert rows['core'].is_problem is False


def test_a_failure_keeps_its_whole_first_line(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the entire reason, because its remedy is usually in the tail.

    chemRxiv's refusal names the bot challenge and then says which sources
    reach the same papers instead. Shortening the reason here would keep the
    half that states the problem and drop the half that solves it, so the
    display shortens it for the column and reprints it whole underneath.
    """
    monkeypatch.setattr(diagnostics, 'load_settings', lambda: {'ncbi_api_key': 'k'})
    monkeypatch.setattr(diagnostics.os, 'environ', {})

    reason = 'refused with 403. ' + 'the remedy is in this tail ' * 12

    def verbose() -> NoReturn:
        """Fail at length, with the useful part last."""
        raise RuntimeError(reason + '\nsecond line dropped')

    monkeypatch.setattr(diagnostics.registry, 'resolve_probe', lambda name: verbose)
    row = diagnostics.provider_status(['pubmed'])[0]
    assert row.detail == reason
    assert len(row.detail) > 120

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


def test_the_cached_pdf_probe_reports_its_size_and_its_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check the billed route, and say what checking it cost.

    A HEAD is billed exactly as a GET is, so this probe spends 100 credits. It
    saves the several-megabyte transfer rather than the charge, and the charge
    is reported so it is never a surprise.
    """
    monkeypatch.setattr(diagnostics.openalex, 'configured_api_key', lambda *_, **__: 'k')
    monkeypatch.setattr(diagnostics.openalex, 'get_work', lambda *_, **__: {'id': 'W1'})
    monkeypatch.setattr(diagnostics.openalex, 'cached_pdf_url',
                        lambda *_, **__: 'https://content.openalex.org/works/W1.pdf')
    monkeypatch.setattr(diagnostics.openalex, 'cached_pdf_available', lambda *_, **__: 1_031_677)

    assert diagnostics._probe_openalex_content() == (
        '1,031,677 byte PDF available; cost 100 credits')

    monkeypatch.setattr(diagnostics.openalex, 'cached_pdf_url', lambda *_, **__: '')
    with pytest.raises(RuntimeError, match='holds no cached PDF'):
        diagnostics._probe_openalex_content()


def test_a_source_without_a_probe_is_reported_as_unchecked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report an unregistered probe rather than implying the source works."""
    monkeypatch.setattr(diagnostics, 'load_settings', lambda: {})
    monkeypatch.setattr(diagnostics.os, 'environ', {})
    monkeypatch.setattr(diagnostics.registry, 'resolve_probe', lambda name: None)

    row = diagnostics.provider_status(['arxiv'])[0]

    assert row.state == diagnostics.NOT_PROBED
    assert row.detail == 'no probe registered'
    assert row.is_problem is False


# Every provider call any probe makes, with a stand-in result. Patching all of
# them keeps this test off the network however the probes are rearranged.
PROBE_CALLS = [
    ('core', 'get_work', {}),
    ('core', 'configured_api_key', 'k'),
    ('unpaywall', 'get_work', {}),
    ('arxiv', 'request_xml', {}),
    ('medrxiv', 'request_json', {}),
    ('biorxiv', 'request_json', {}),
    ('chemrxiv', 'request_json', {}),
    ('pubmed', 'esearch', ([], '')),
    ('pubmed', 'configured_api_key', 'k'),
    ('pubmed', 'configured_email', ''),
    ('crossref', 'work_by_doi', {}),
    ('crossref', 'resolve_email', ''),
    ('openalex', 'get_work', {'id': 'https://openalex.org/W1'}),
    ('openalex', 'configured_api_key', 'k'),
    ('openalex', 'budget_status', None),
    ('openalex', 'cached_pdf_url', 'https://content.openalex.org/works/W1.pdf'),
    ('openalex', 'cached_pdf_available', 1024),
    ('elsevier', 'check_api_key', True),
    ('elsevier', 'configured_api_key', 'k'),
    ('elsevier', 'quota_status', None),
]
# The call that proves each source was actually reached.
PROBE_TARGETS = [
    ('core', 'core', 'get_work'),
    ('unpaywall', 'unpaywall', 'get_work'),
    ('arxiv', 'arxiv', 'request_xml'),
    ('medrxiv', 'medrxiv', 'request_json'),
    ('biorxiv', 'biorxiv', 'request_json'),
    ('chemrxiv', 'chemrxiv', 'request_json'),
    ('pubmed', 'pubmed', 'esearch'),
    ('crossref', 'crossref', 'work_by_doi'),
    ('openalex', 'openalex', 'get_work'),
    ('elsevier', 'elsevier', 'check_api_key'),
    ('openalex-content', 'openalex', 'cached_pdf_available'),
]


@pytest.mark.parametrize(('source', 'module_name', 'function_name'), PROBE_TARGETS)
def test_each_probe_calls_its_provider_with_arguments_it_accepts(
    source: str,
    module_name: str,
    function_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bind each probe's call against the real signature it is aimed at.

    Replacing a provider function with a permissive stub makes a probe test
    pass whatever the probe passes, which is how ``core.search_page`` came to
    be called with a ``page_size`` argument it does not take: the fault only
    appeared once someone configured a CORE key and the probe actually ran.
    Binding against the genuine signature catches that without a network call.

    Parameters
    ----------
    source : str
        Registry source whose probe is under test.
    module_name : str
        Provider module attribute on the diagnostics module.
    function_name : str
        Function that proves the provider was reached.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to substitute every provider call.
    """
    reached: list[str] = []

    def substitute(module: object, name: str, result: object) -> None:
        """Replace one provider call with a signature-checked stand-in."""
        signature = inspect.signature(getattr(module, name))

        def checked(*args: object, **kwargs: object) -> object:
            """Reject a call the real function could not have accepted."""
            signature.bind(*args, **kwargs)
            reached.append(name)
            return result

        monkeypatch.setattr(module, name, checked)

    for patched_module, patched_function, result in PROBE_CALLS:
        substitute(getattr(diagnostics, patched_module), patched_function, result)

    probe = registry.resolve_probe(source)
    assert probe is not None
    probe()
    assert function_name in reached, f'{source} probe never called {function_name}'


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

    # The providers whose probes report only that they answered are covered by
    # test_each_probe_calls_its_provider_with_arguments_it_accepts, which
    # exercises every one of them without a network call.
    monkeypatch.setattr(diagnostics.core, 'get_work', lambda *_, **__: None)
    monkeypatch.setattr(diagnostics.core, 'configured_api_key', lambda *_, **__: 'k')
    # A withdrawn record is an answer: CORE responded, which is what is asked.
    assert diagnostics._probe_core() == 'answered'


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


def test_cached_pdf_available_checks_without_transferring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Confirm the cached PDF is servable using a HEAD, and report its size.

    The HEAD is billed exactly as a GET is, so this saves several megabytes of
    transfer rather than the charge. It still has to spend the budget check and
    record what the response says is left.
    """
    from tests.doubles import FakeResponse

    calls: list[dict[str, object]] = []

    def fake_head(url: str, **kwargs: object) -> FakeResponse:
        """Record the HEAD and answer as OpenAlex does."""
        calls.append({'url': url, **kwargs})
        return FakeResponse(headers={'Content-Type': 'application/pdf',
                                     'Content-Length': '1031677',
                                     diagnostics.openalex.BUDGET_REMAINING_HEADER: '9800'})

    monkeypatch.setattr(diagnostics.openalex.requests, 'head', fake_head)
    size = diagnostics.openalex.cached_pdf_available(
        'https://content.openalex.org/works/W1.pdf', 'openalex-key')

    assert size == 1_031_677
    assert calls[0]['params'] == {'api_key': 'openalex-key'}
    # The response's budget figure is remembered, as any content response is.
    assert diagnostics.openalex.budget_status().remaining == 9800


def test_cached_pdf_available_reports_what_went_wrong(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuse without a key, on a rejection, and on a non-PDF answer."""
    from tests.doubles import FakeResponse

    with pytest.raises(ValueError, match='require an API key'):
        diagnostics.openalex.cached_pdf_available('https://content.example/W1.pdf', '  ')

    monkeypatch.setattr(diagnostics.openalex.requests, 'head',
                        lambda *_, **__: FakeResponse(status_code=401))
    with pytest.raises(RuntimeError, match='rejected the API key'):
        diagnostics.openalex.cached_pdf_available('https://content.example/W1.pdf', 'k')

    monkeypatch.setattr(diagnostics.openalex.requests, 'head',
                        lambda *_, **__: FakeResponse(status_code=503))
    with pytest.raises(RuntimeError, match='refused the cached PDF with 503'):
        diagnostics.openalex.cached_pdf_available('https://content.example/W1.pdf', 'k')

    monkeypatch.setattr(diagnostics.openalex.requests, 'head',
                        lambda *_, **__: FakeResponse(headers={'Content-Type': 'text/html'}))
    with pytest.raises(RuntimeError, match='served text/html rather than a PDF'):
        diagnostics.openalex.cached_pdf_available('https://content.example/W1.pdf', 'k')

    # A PDF whose length the response does not declare is still available.
    monkeypatch.setattr(diagnostics.openalex.requests, 'head',
                        lambda *_, **__: FakeResponse(headers={'Content-Type': 'application/pdf'}))
    assert diagnostics.openalex.cached_pdf_available('https://content.example/W1.pdf', 'k') == 0

    monkeypatch.setattr(diagnostics.openalex.requests, 'head',
                        lambda *_, **__: FakeResponse(status_code=500))
    with pytest.raises(RuntimeError, match='refused the cached PDF with 500'):
        diagnostics.openalex.cached_pdf_available('https://content.example/W1.pdf', 'k')


def test_a_source_with_no_probe_target_resolves_to_none() -> None:
    """Allow a source to declare no probe, and report it as unchecked."""
    from dataclasses import replace

    assert _resolve(replace(registry.SOURCES['arxiv'], probe='')) is None
