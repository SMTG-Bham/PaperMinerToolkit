"""Offline checks for the provider probing command and its script interface."""

from __future__ import annotations

import json
from urllib.parse import quote

import click
import pytest
from click.testing import CliRunner

from paperminertoolkit import probe_cli
from paperminertoolkit.providers import registry
from paperminertoolkit.workflows import diagnostics


@pytest.fixture(autouse=True)
def offline_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep reports independent of the machine's credentials."""
    monkeypatch.setattr(diagnostics, 'load_settings', lambda: {})
    for entry in registry.SOURCES.values():
        if entry.credential_env:
            monkeypatch.delenv(entry.credential_env, raising=False)
    monkeypatch.delenv('NCBI_EMAIL', raising=False)


def test_help_lists_registry_sources_and_probe_cost() -> None:
    """Make all supported sources and the metered probe discoverable."""
    result = CliRunner().invoke(probe_cli.probe, ['--help'])
    assert result.exit_code == 0
    assert all(name in result.output for name in registry.SOURCES)
    assert '100 OpenAlex credits' in result.output


@pytest.mark.parametrize(('arguments', 'expected'), [
    ([], None),
    (['--source', 'ARXIV', '--source', 'pubmed', '--source', 'arxiv'], ['pubmed', 'arxiv']),
    (['--source', 'arxiv', '--source', 'all'], None),
])
def test_provider_selection_is_deduplicated_in_registry_order(
    monkeypatch: pytest.MonkeyPatch, arguments: list[str], expected: list[str] | None,
) -> None:
    """Request each selected provider only once and default to all sources."""
    calls: list[tuple[list[str] | None, bool]] = []

    def fake_status(names: list[str] | None, probe: bool) -> list[diagnostics.ProviderStatus]:
        """Record the diagnostic request without making network calls."""
        calls.append((names, probe))
        return []

    monkeypatch.setattr(diagnostics, 'provider_status', fake_status)
    result = CliRunner().invoke(probe_cli.probe, arguments)
    assert result.exit_code == 0
    assert calls == [(expected, True)]


def test_unknown_source_is_rejected_before_probing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate a source name before any provider can be contacted."""
    def forbidden(*args: object, **kwargs: object) -> None:
        """Fail if diagnostics is reached for an invalid command."""
        pytest.fail('invalid sources must not reach diagnostics')

    monkeypatch.setattr(diagnostics, 'provider_status', forbidden)
    result = CliRunner().invoke(probe_cli.probe, ['--source', 'unknown'])
    assert result.exit_code == 2
    assert 'Invalid value' in result.output
    # The shared helper also validates legacy callers without Click choices.
    with pytest.raises(click.BadParameter, match='Unknown provider'):
        probe_cli.report_provider_status(False, ('unknown',))


def test_offline_check_never_resolves_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the real diagnostics flow while refusing all network targets."""
    def forbidden(name: str) -> None:
        """Fail if an offline check attempts to resolve a probe."""
        pytest.fail(f'offline check resolved {name}')

    monkeypatch.setattr(registry, 'resolve_probe', forbidden)
    result = CliRunner().invoke(probe_cli.probe, ['--no-probe', '--json'])
    assert result.exit_code == 0
    report = json.loads(result.stdout)
    assert report['ok'] is True
    rows = {row['name']: row for row in report['providers']}
    assert set(rows) == set(registry.SOURCES)
    assert rows['core']['state'] == diagnostics.NOT_SET_UP
    assert rows['arxiv']['state'] == diagnostics.NOT_PROBED
    assert rows['arxiv']['detail'] == 'no credential needed'


def test_json_report_stays_parseable_when_clients_print(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep notices out of JSON and represent missing setup without failing."""
    def noisy_status(*args: object, **kwargs: object) -> list[diagnostics.ProviderStatus]:
        """Simulate a client emitting notices on both output streams."""
        click.echo('provider notice')
        click.echo('provider warning', err=True)
        return [diagnostics.ProviderStatus('core', 'CORE', 'CORE_API_KEY',
                                           diagnostics.NOT_SET_UP, 'run pmt config core-key')]

    monkeypatch.setattr(diagnostics, 'provider_status', noisy_status)
    result = CliRunner().invoke(probe_cli.probe, ['--json'])
    assert result.exit_code == 0
    assert result.stderr == ''
    report = json.loads(result.stdout)
    assert report['ok'] is True
    assert report['providers'][0]['credential'] == 'CORE_API_KEY'
    assert report['providers'][0]['state'] == diagnostics.NOT_SET_UP


@pytest.mark.parametrize('json_output', [False, True])
def test_provider_failure_returns_nonzero_with_full_reason(
    monkeypatch: pytest.MonkeyPatch, json_output: bool,
) -> None:
    """Preserve actionable failures in human and machine-readable reports."""
    reason = 'Service refused the request. ' * 5 + 'Try again after the provider recovers.'
    row = diagnostics.ProviderStatus('arxiv', 'arXiv', '', diagnostics.NOT_RESPONDING,
                                     reason, 0.4)
    monkeypatch.setattr(diagnostics, 'provider_status', lambda *args, **kwargs: [row])
    result = CliRunner().invoke(probe_cli.probe, ['--json'] if json_output else [])
    assert result.exit_code == 1
    if json_output:
        report = json.loads(result.stdout)
        assert report['ok'] is False
        assert report['providers'][0]['detail'] == reason
        assert report['providers'][0]['seconds'] == 0.4
        assert result.stderr == ''
    else:
        assert reason in result.stderr
        assert '1 configured provider(s) not responding: arXiv' in result.stderr
        assert '0.4s' in result.stdout


@pytest.mark.parametrize('json_output', [False, True])
def test_failures_never_print_configured_credential_values(
    monkeypatch: pytest.MonkeyPatch, json_output: bool,
) -> None:
    """Redact raw and URL-encoded credentials reflected in provider failures."""
    secret = 'example+secret/key'
    contact = 'private@example.test'
    monkeypatch.setattr(diagnostics, 'load_settings', lambda: {'core_api_key': secret})
    monkeypatch.setenv('NCBI_EMAIL', contact)

    def refuse() -> str:
        """Echo an authenticated URL the way an HTTP error can."""
        click.echo(f'Connecting with {secret}')
        click.echo(f'Contact: {contact}', err=True)
        raise RuntimeError(f'request failed: api_key={quote(secret, safe="")} '
                           f'contact={contact}, token={secret}')

    monkeypatch.setattr(registry, 'resolve_probe', lambda name: refuse)
    arguments = ['--source', 'core', *(['--json'] if json_output else [])]
    result = CliRunner().invoke(probe_cli.probe, arguments)
    assert result.exit_code == 1
    assert secret not in result.output
    assert quote(secret, safe='') not in result.output
    assert contact not in result.output
    assert '[redacted]' in result.output
    assert 'CORE_API_KEY' in result.output


def test_configuration_errors_use_click_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Present unreadable settings without a traceback or malformed JSON."""
    def invalid_settings() -> dict[str, object]:
        """Simulate the diagnostic settings loader rejecting a file."""
        raise RuntimeError('Could not read settings')

    monkeypatch.setattr(diagnostics, 'load_settings', invalid_settings)
    result = CliRunner().invoke(probe_cli.probe, ['--no-probe', '--json'])
    assert result.exit_code == 1
    assert result.stdout == ''
    assert result.stderr == 'Error: Could not read settings\n'
