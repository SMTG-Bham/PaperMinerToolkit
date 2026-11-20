"""Provider readiness checks exposed through the ``pmt probe`` command."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, replace
from io import StringIO
from urllib.parse import quote, quote_plus

import click

from paperminertoolkit.providers import registry
from paperminertoolkit.workflows import diagnostics


def _credential_values(settings: Mapping[str, object]) -> list[str]:
    """Collect configured provider credentials solely for output redaction."""
    values = set()
    for entry in registry.SOURCES.values():
        for value in (settings.get(entry.credential), os.environ.get(entry.credential_env)):
            if isinstance(value, str) and value.strip():
                values.add(value.strip())
    # PubMed sends a separate contact address alongside its optional API key.
    for value in (settings.get('ncbi_email'), os.environ.get('NCBI_EMAIL')):
        if isinstance(value, str) and value.strip():
            values.add(value.strip())
    return sorted({variant for value in values
                   for variant in (value, quote(value, safe=''), quote_plus(value))},
                  key=len, reverse=True)


def _redact(text: str, credentials: list[str]) -> str:
    """Remove credential values that provider errors may include in URLs."""
    for value in credentials:
        text = text.replace(value, '[redacted]')
    return text


def report_provider_status(no_probe: bool, sources: tuple[str, ...],
                           json_output: bool = False) -> None:
    """Report provider readiness for both the probe and configuration commands.

    Parameters
    ----------
    no_probe : bool
        Inspect local configuration without making provider requests.
    sources : tuple[str, ...]
        Registry names to inspect. An empty tuple or ``all`` selects all.
    json_output : bool, default=False
        Print a JSON object with ``providers`` rows and an ``ok`` boolean.

    Raises
    ------
    click.BadParameter
        If a requested provider is not registered.
    click.ClickException
        If configuration cannot be read.
    SystemExit
        With code 1 when a provider reports a reachability failure. Providers
        missing required credentials are reported without failing the command.
    """
    requested = {source.lower() for source in sources}
    unknown = requested - registry.SOURCES.keys() - {'all'}
    if unknown:
        raise click.BadParameter(f'Unknown provider(s): {", ".join(sorted(unknown))}',
                                 param_hint='--source')
    names = (None if not requested or 'all' in requested else
             [name for name in registry.SOURCES if name in requested])
    credentials: list[str] = []
    try:
        # Provider clients can print notices or URLs. The report contains their
        # outcome, while silencing that chatter keeps JSON usable and secrets
        # out of both output streams.
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            credentials = _credential_values(diagnostics.load_settings())
            rows = diagnostics.provider_status(names, probe=not no_probe)
    except (RuntimeError, ValueError) as error:
        raise click.ClickException(_redact(str(error), credentials)) from error
    rows = [replace(row, detail=_redact(row.detail, credentials)) for row in rows]
    broken = [row for row in rows if row.is_problem]
    if json_output:
        click.echo(json.dumps({'providers': [asdict(row) for row in rows],
                               'ok': not broken}, indent=2))
    else:
        width = max((len(row.label) for row in rows), default=0)
        for row in rows:
            took = f' {row.seconds:.1f}s' if row.seconds else ''
            credential = row.credential or '-'
            detail = row.detail if len(row.detail) <= 76 else f'{row.detail[:73]}...'
            click.echo(f'{row.label:<{width}}  {row.state:<15} {credential:<18} {detail}{took}')
        if broken:
            for row in broken:
                click.echo(f'\n{row.label}: {row.detail}', err=True)
            click.echo(f'\n{len(broken)} configured provider(s) not responding: '
                       f'{", ".join(row.label for row in broken)}', err=True)
    if broken:
        raise SystemExit(1)


@click.command('probe')
@click.option('--source', 'sources', multiple=True,
              type=click.Choice(['all', *registry.SOURCES], case_sensitive=False),
              default=('all',), show_default=True,
              help='Provider to check. Repeat to choose several.')
@click.option('--no-probe', is_flag=True,
              help='Report configuration without making any requests.')
@click.option('--json', 'json_output', is_flag=True,
              help='Emit machine-readable JSON with provider rows and an ok flag.')
def probe(no_probe: bool, sources: tuple[str, ...], json_output: bool) -> None:
    """Check paper provider configuration and reachability.

    Requests respect each provider's rate limiter. Checking openalex-content
    costs 100 OpenAlex credits when its key is configured. Use --no-probe for
    an offline check. Missing credentials are reported without failing; a
    provider that is not responding makes the command exit with status 1.
    """
    report_provider_status(no_probe, sources, json_output)
