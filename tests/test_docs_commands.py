"""Keep every ``pmt`` command line shown in the documentation valid.

The guides, the README and the example notebooks show many complete ``pmt``
invocations. They drift silently when an option is renamed, a choice value is
dropped or a command moves into a group, so this module extracts each one from
the shell code blocks and checks it against the live Click tree: the command
path must resolve, every option must exist on that command, a choice value
must be one the option accepts, and there must be no more positional values
than the command takes. Nothing is executed and nothing touches the network.
"""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import pytest

from paperminertoolkit import cli

ROOT = Path(__file__).parents[1]
DOC_FILES = (ROOT / 'README.md', *sorted((ROOT / 'docs').rglob('*.md')))
NOTEBOOKS = tuple(sorted((ROOT / 'docs').rglob('*.ipynb')))

#: Fence languages whose lines are read as shell commands.
SHELL_LANGUAGES = {'', 'bash', 'console', 'sh', 'shell', 'shell-session', 'zsh'}

#: Doc lines that are deliberately not valid commands, keyed by
#: ``(path relative to the repository, command line)``, with the reason.
ALLOWLIST: dict[tuple[str, str], str] = {}

FENCE = re.compile(r'^\s*(?P<marker>`{3,}|~{3,})(?P<info>.*)$')
PLACEHOLDER = re.compile(r'<[^<>\s]+>')
CONTROL = {'|', '||', '&&', ';', '&', '(', ')', '$('}
ASSIGNMENT = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*=')


@dataclass(frozen=True)
class Invocation:
    """One ``pmt`` command line found in the documentation.

    Attributes
    ----------
    source : str
        File the line came from, relative to the repository root.
    line : int
        Line number of the command's first physical line; for a notebook, the
        cell index.
    text : str
        The command with continuations joined and any prompt removed.
    """

    source: str
    line: int
    text: str


def _fence_language(info: str) -> str | None:
    """Return the shell-relevant language of a fence's info string.

    Parameters
    ----------
    info : str
        Text after the opening fence marker.

    Returns
    -------
    str or None
        Lower-case language name, ``''`` for a bare fence, or ``None`` for a
        MyST directive that is not a code block.
    """
    info = info.strip()
    if info.startswith('{'):
        directive, _, argument = info[1:].partition('}')
        if directive not in {'code-block', 'code', 'sourcecode'}:
            return None
        info = argument.strip()
    return info.split(maxsplit=1)[0].lower() if info else ''


def _shell_blocks(lines: list[str]) -> Iterator[tuple[int, list[str], str]]:
    """Yield the shell code blocks of a Markdown document.

    Parameters
    ----------
    lines : list of str
        Document lines.

    Yields
    ------
    tuple of int, list of str and str
        One-based line number of the block's first line, its lines, and its
        language.
    """
    index = 0
    while index < len(lines):
        opening = FENCE.match(lines[index])
        index += 1
        if opening is None:
            continue
        marker = opening.group('marker')
        language = _fence_language(opening.group('info'))
        start = index
        while index < len(lines):
            closing = FENCE.match(lines[index])
            if (closing is not None and closing.group('marker')[0] == marker[0]
                    and len(closing.group('marker')) >= len(marker)
                    and not closing.group('info').strip()):
                break
            index += 1
        if language in SHELL_LANGUAGES:
            yield start + 1, lines[start:index], language
        index += 1


def _logical_lines(lines: list[str], first: int) -> Iterator[tuple[int, str]]:
    """Join backslash continuations into logical lines.

    Parameters
    ----------
    lines : list of str
        Physical lines of one code block.
    first : int
        Line number of ``lines[0]``.

    Yields
    ------
    tuple of int and str
        Line number where the logical line starts, and the joined text.
    """
    buffer: list[str] = []
    start = first
    for offset, line in enumerate(lines):
        if not buffer:
            start = first + offset
        stripped = line.rstrip()
        if stripped.endswith('\\'):
            buffer.append(stripped[:-1].strip())
            continue
        buffer.append(stripped.strip())
        yield start, ' '.join(part for part in buffer if part)
        buffer = []
    if buffer:
        yield start, ' '.join(part for part in buffer if part)


def _commands(line: str, language: str) -> Iterator[str]:
    """Yield the ``pmt`` commands on one logical shell line.

    Lines in a ``console`` block are commands only after a ``$`` prompt, so
    output lines are skipped there. Elsewhere a line counts only when ``pmt``
    is the first word of a pipeline segment, which leaves prose-like output
    such as ``run pmt config ncbi-key`` alone.

    Parameters
    ----------
    line : str
        Joined shell line.
    language : str
        Language of the enclosing code block.

    Yields
    ------
    str
        Each command, from ``pmt`` up to the next shell control operator.
    """
    line = line.strip()
    if line.startswith('!'):
        line = line[1:].lstrip()
    if line.startswith('$ '):
        line = line[2:]
    elif language in {'console', 'shell-session'}:
        return
    if 'pmt' not in line or line.startswith('#'):
        return
    line = PLACEHOLDER.sub('PLACEHOLDER', line)
    lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:
        return
    segment: list[str] = []
    redirected = False
    for token in [*tokens, ';']:
        if redirected:
            redirected = False
            continue
        if set(token) <= set('<>&') and set(token) & set('<>'):
            # Drop a redirection and its target, and a file descriptor before it.
            if segment and segment[-1] in {'1', '2'}:
                segment.pop()
            redirected = True
            continue
        if token in CONTROL:
            while segment and ASSIGNMENT.match(segment[0]):
                segment.pop(0)
            if segment and segment[0] == 'pmt':
                yield shlex.join(segment)
            segment = []
        else:
            segment.append(token)


def _markdown_invocations(source: str, text: str, offset: int = 0) -> Iterator[Invocation]:
    """Yield the ``pmt`` invocations in the shell blocks of Markdown text.

    Parameters
    ----------
    source : str
        Label for the text's origin.
    text : str
        Markdown source.
    offset : int, optional
        Value added to every line number, for text embedded in another file.

    Yields
    ------
    Invocation
        Each command found.
    """
    for first, block, language in _shell_blocks(text.splitlines()):
        for number, line in _logical_lines(block, first):
            for command in _commands(line, language):
                yield Invocation(source, number + offset, command)


def _notebook_invocations(source: str, cells: list[dict[str, Any]]) -> Iterator[Invocation]:
    """Yield the ``pmt`` invocations in a notebook's cells.

    ``%%bash`` and ``%%sh`` cells are read as shell, ``!pmt`` lines in other
    code cells as shell escapes, and Markdown cells by their fenced blocks.
    Commands built at run time, for example inside ``subprocess`` argument
    lists, are not literal command lines and are skipped.

    Parameters
    ----------
    source : str
        Label for the notebook.
    cells : list of dict
        The notebook's ``cells`` array.

    Yields
    ------
    Invocation
        Each command found; ``line`` is the cell index.
    """
    for index, cell in enumerate(cells):
        text = ''.join(cell['source'])
        if cell['cell_type'] == 'markdown':
            for invocation in _markdown_invocations(source, text):
                yield Invocation(source, index, invocation.text)
            continue
        if cell['cell_type'] != 'code':
            continue
        lines = text.splitlines()
        shell = bool(lines) and lines[0].startswith(('%%bash', '%%sh'))
        body = lines[1:] if shell else lines
        for _, line in _logical_lines(body, 1):
            if shell or line.lstrip().startswith('!'):
                for command in _commands(line, 'bash'):
                    yield Invocation(source, index, command)


def _invocations() -> list[Invocation]:
    """Collect every documented ``pmt`` invocation.

    Returns
    -------
    list of Invocation
        Invocations from the README, the Markdown docs, and the notebooks.
    """
    found: list[Invocation] = []
    for path in DOC_FILES:
        found.extend(_markdown_invocations(str(path.relative_to(ROOT)),
                                           path.read_text(encoding='utf-8')))
    for path in NOTEBOOKS:
        cells = json.loads(path.read_text(encoding='utf-8'))['cells']
        found.extend(_notebook_invocations(str(path.relative_to(ROOT)), cells))
    return found


def _check_choice(parameter: click.Parameter, value: str, where: str) -> list[str]:
    """Check one literal value against a parameter's ``Choice`` type.

    Parameters
    ----------
    parameter : click.Parameter
        Option or argument receiving the value.
    value : str
        Value written in the documentation.
    where : str
        How to name the parameter in a problem message.

    Returns
    -------
    list of str
        A problem message, or nothing when the value is acceptable or not a
        literal.
    """
    kind = parameter.type
    if not isinstance(kind, click.Choice) or '$' in value or 'PLACEHOLDER' in value:
        return []
    choices = [str(choice) for choice in kind.choices]
    if kind.case_sensitive:
        accepted = value in choices
    else:
        accepted = value.casefold() in {choice.casefold() for choice in choices}
    return [] if accepted else [f'{where} does not accept {value!r} (choices: {", ".join(choices)})']


def _problems(command_line: str) -> list[str]:
    """Return what is wrong with one ``pmt`` command line.

    Parameters
    ----------
    command_line : str
        Shell-quoted command starting with ``pmt``.

    Returns
    -------
    list of str
        Problem messages; empty when the command is valid.
    """
    tokens = shlex.split(command_line)[1:]
    command: click.Command = cli.main
    path = ['pmt']
    positional: list[str] = []
    problems: list[str] = []
    stops = False
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        options = {name: parameter for parameter in command.params
                   if isinstance(parameter, click.Option)
                   for name in (*parameter.opts, *parameter.secondary_opts)}
        if token == '--':
            positional.extend(tokens[index:])
            break
        if token.startswith('-') and len(token) > 1 and not token[1:].isdigit():
            name, has_value, value = token.partition('=')
            if name == '--help':
                stops = True
                continue
            option = options.get(name)
            if option is None:
                problems.append(f'{" ".join(path)} has no option {name}')
                continue
            if option.is_eager:
                stops = True
            if option.is_flag or option.count:
                continue
            values = [value] if has_value else tokens[index:index + option.nargs]
            if not has_value:
                index += option.nargs
            for item in values:
                problems.extend(_check_choice(option, item, f'{" ".join(path)} {name}'))
            continue
        if isinstance(command, click.Group) and not positional:
            child = command.get_command(click.Context(command), token)
            if child is None:
                problems.append(f'{" ".join(path)} has no command {token!r}')
                return problems
            command = child
            path.append(token)
            continue
        positional.append(token)
    if stops or isinstance(command, click.Group):
        return problems
    arguments = [parameter for parameter in command.params if isinstance(parameter, click.Argument)]
    if any(argument.nargs == -1 for argument in arguments):
        return problems
    if '...' in positional:
        return problems
    if len(positional) > sum(argument.nargs for argument in arguments):
        problems.append(f'{" ".join(path)} takes {len(arguments)} argument(s), '
                        f'got {len(positional)}: {positional}')
    required = [argument for argument in arguments if argument.required]
    if len(positional) < sum(argument.nargs for argument in required):
        problems.append(f'{" ".join(path)} is missing required arguments: '
                        f'{[argument.name for argument in required[len(positional):]]}')
    for argument, value in zip(arguments, positional, strict=False):
        problems.extend(_check_choice(argument, value, f'{" ".join(path)} {argument.name}'))
    return problems


INVOCATIONS = _invocations()


def test_documentation_commands_are_found() -> None:
    """Fail loudly if extraction stops finding the documented commands."""
    sources = {invocation.source for invocation in INVOCATIONS}
    assert len(INVOCATIONS) > 150
    assert {'README.md', 'docs/workflow/corpus.md'} <= sources
    assert any(source.endswith('.ipynb') for source in sources)
    assert any('figures' in invocation.text for invocation in INVOCATIONS)


def test_extraction_handles_prompts_continuations_and_output() -> None:
    """Join continuations, drop prompts and ignore output and comments."""
    text = '\n'.join([
        '```console',
        '$ pmt corpus stats papers.db \\',
        '    --json',
        'pmt status papers.db',
        '```',
        '```bash',
        '# pmt not-a-command',
        'printf x | pmt config core-key --value-stdin  # trailing comment',
        'Crossref ok; run pmt config crossref-email',
        'pmt scrape <db> <recipe> --dry-run > plan.json',
        '```',
        '```{toctree}',
        'pmt ignored',
        '```',
    ])
    found = [(invocation.line, invocation.text) for invocation in _markdown_invocations('x', text)]
    assert found == [
        (2, 'pmt corpus stats papers.db --json'),
        (8, 'pmt config core-key --value-stdin'),
        (10, 'pmt scrape PLACEHOLDER PLACEHOLDER --dry-run'),
    ]


def test_notebook_extraction_reads_shell_cells_and_escapes_only() -> None:
    """Read ``%%bash`` cells and ``!pmt`` escapes but not run-time commands."""
    cells = [
        {'cell_type': 'code', 'source': ['%%bash\n', 'set -euo pipefail\n', 'pmt status \"$PMT_DB\"']},
        {'cell_type': 'code', 'source': ['import subprocess\n', "subprocess.run(['pmt', 'status'])\n",
                                         '!pmt corpus stats papers.db --json']},
        {'cell_type': 'markdown', 'source': ['```bash\n', 'pmt topics models papers.db\n', '```']},
        {'cell_type': 'raw', 'source': ['pmt ignored']},
    ]
    found = [(invocation.line, invocation.text) for invocation in _notebook_invocations('nb', cells)]
    assert found == [
        (0, "pmt status '$PMT_DB'"),
        (1, 'pmt corpus stats papers.db --json'),
        (2, 'pmt topics models papers.db'),
    ]


@pytest.mark.parametrize(
    ('command_line', 'expected'),
    [
        ('pmt corpus stats papers.db --json', []),
        ('pmt --version', []),
        ('pmt corpus nope papers.db', ["pmt corpus has no command 'nope'"]),
        ('pmt corpus stats papers.db --nope', ['pmt corpus stats has no option --nope']),
        ('pmt corpus stats a.db b.db', ["pmt corpus stats takes 1 argument(s), got 2: ['a.db', 'b.db']"]),
    ],
)
def test_problems_reports_unknown_commands_options_and_arguments(command_line: str,
                                                                 expected: list[str]) -> None:
    """Check the validator against known good and bad command lines."""
    assert _problems(command_line) == expected


@pytest.mark.parametrize('invocation', INVOCATIONS,
                         ids=[f'{item.source}:{item.line}' for item in INVOCATIONS])
def test_documented_command_is_valid(invocation: Invocation) -> None:
    """Require each documented command to match the installed CLI."""
    if (invocation.source, invocation.text) in ALLOWLIST:
        pytest.skip(ALLOWLIST[invocation.source, invocation.text])
    assert _problems(invocation.text) == [], invocation.text


def test_allowlist_entries_are_still_present() -> None:
    """Drop allowlist entries whose documentation line has gone."""
    present = {(invocation.source, invocation.text) for invocation in INVOCATIONS}
    assert set(ALLOWLIST) <= present
