"""Keep every ``pmt`` command machine-readable and documented for agents.

The test walks the live command tree. A command must either accept
``--json`` (or emit JSON by default) or be listed in ``EXEMPT`` with the
reason it has none, and every JSON command must have a row in the JSON table
of the agent guide. Remove an entry from ``EXEMPT`` when its command gains
JSON output; the test then insists on the documentation row.
"""

from __future__ import annotations

import re
from pathlib import Path

import click
import pytest

from paperminertoolkit import cli
from paperminertoolkit._agent import command_effects

AGENT_GUIDE = Path(__file__).parents[1] / 'docs' / 'workflow' / 'agents.md'

#: Commands that print JSON without a ``--json`` flag.
JSON_BY_DEFAULT = {'schema', 'recipe show'}

#: Commands without JSON output, and why. Keep this list shrinking.
EXEMPT: dict[str, str] = {
    'validate gui': 'Starts an interactive local review server.',
    'recipe prompt': 'Renders a plain-text prompt by design.',
    'import author': 'Pending #96.',
    'filter regex': 'Pending #96.',
    'filter topic': 'Pending #96.',
    'filter reset': 'Pending #96.',
    'topics train': 'Pending #96.',
    'topics compare': 'Pending #96.',
    'topics show': 'Pending #96.',
    'topics name': 'Pending #96.',
    'topics predict': 'Pending #96.',
    'topics trends': 'Pending #96.',
    'topics store': 'Pending #96.',
    'topics models': 'Pending #96.',
    'entities corpus': 'Pending #96.',
    'entities text': 'Pending #96.',
    'validate template': 'Pending #96.',
    'scrape': 'Pending #97.',
    'store': 'Pending #97.',
    'import pdfs': 'Pending #97.',
    'reset': 'Pending #97.',
    'config model': 'Pending #104.',
    'config elsevier-key': 'Pending #104.',
    'config core-key': 'Pending #104.',
    'config core-rate': 'Pending #104.',
    'config unpaywall-email': 'Pending #104.',
    'config crossref-email': 'Pending #104.',
    'config openalex-key': 'Pending #104.',
    'config ncbi-key': 'Pending #104.',
    'config ncbi-email': 'Pending #104.',
    'config openai-key': 'Pending #104.',
    'config anthropic-key': 'Pending #104.',
}


def _leaf_commands(command: click.Command, path: tuple[str, ...] = ()) -> list[tuple[str, click.Command]]:
    """Collect every runnable command below ``command`` with its path after ``pmt``."""
    if not isinstance(command, click.Group):
        return [(' '.join(path), command)]
    context = click.Context(command)
    leaves = []
    for name in command.list_commands(context):
        child = command.get_command(context, name)
        assert child is not None
        leaves.extend(_leaf_commands(child, (*path, name)))
    return leaves


def _json_table_commands() -> str:
    """Return the first column of the guide's JSON table as one string."""
    text = AGENT_GUIDE.read_text(encoding='utf-8')
    section = text.split('## JSON and exit-code contract', 1)[1]
    table = section.split('| Exit code |', 1)[0]
    return '\n'.join(line.split('|')[1] for line in table.splitlines()
                     if line.startswith('| `'))


LEAVES = _leaf_commands(cli.main)


def _accepts_json_flag(command: click.Command) -> bool:
    """Return whether the command declares the shared ``--json`` flag."""
    return any('--json' in getattr(parameter, 'opts', ()) for parameter in command.params)


@pytest.mark.parametrize(('path', 'command'), LEAVES, ids=[path for path, _ in LEAVES])
def test_command_is_machine_readable_or_exempt(path: str, command: click.Command) -> None:
    """Require JSON output unless the command is explicitly exempt."""
    supports_json = _accepts_json_flag(command) or path in JSON_BY_DEFAULT
    if path in EXEMPT:
        assert not supports_json, f'{path} now supports JSON; remove it from EXEMPT'
    else:
        assert supports_json, f'{path} has no --json; add it or list it in EXEMPT with a reason'
    assert command_effects(command), f'{path} declares no effects'


@pytest.mark.parametrize(('path', 'command'), LEAVES, ids=[path for path, _ in LEAVES])
def test_json_command_is_documented_for_agents(path: str, command: click.Command) -> None:
    """Require a row in the agent guide's JSON table for every JSON command."""
    if path in EXEMPT:
        pytest.skip(f'{path} has no JSON output: {EXEMPT[path]}')
    assert re.search(r'`' + re.escape(path) + r'[ `]', _json_table_commands()), (
        f'{path} is missing from the JSON table in {AGENT_GUIDE.name}'
    )


def test_exemptions_name_real_commands() -> None:
    """Keep stale exemptions from hiding renamed commands."""
    assert set(EXEMPT) <= {path for path, _ in LEAVES}
