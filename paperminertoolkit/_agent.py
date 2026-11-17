"""Machine-readable command metadata for agents that drive the ``pmt`` CLI.

Each leaf command declares its side effects with :func:`agent_meta`. The
``pmt schema`` output carries them, so a caller can tell which commands make
network requests, call a language model, or change the corpus before running
them.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import PurePath
from typing import Any

import click

#: Side-effect vocabulary. Each command lists the effects it can have with its
#: default options; flags such as ``--no-probe`` or ``--dry-run`` can remove some.
EFFECTS: dict[str, str] = {
    'read_only': 'Reads local files or the corpus without changing anything.',
    'migrates_corpus': ('Opens the corpus normally, which upgrades an older corpus schema '
                        'in place but changes no paper data.'),
    'network': 'Makes requests to external providers or model hubs.',
    'paid_model': ('Calls the configured language model; hosted providers bill these '
                   'requests.'),
    'writes_corpus': 'Changes papers, assets, statuses, or filters in the corpus database.',
    'writes_files': ('Writes files outside the corpus, such as CSV, JSON, images, or model '
                     'artifacts.'),
    'writes_config': 'Changes the saved PaperMinerToolkit settings file.',
    'prompts': 'Can wait for input on stdin unless the documented options are supplied.',
    'starts_server': 'Runs a local server until it is interrupted.',
}

# These effects describe a change; ``read_only`` cannot accompany them.
_WRITING_EFFECTS = frozenset({'migrates_corpus', 'writes_corpus', 'writes_files',
                              'writes_config'})


def agent_meta(*, effects: Iterable[str]) -> Callable[[click.Command], click.Command]:
    """Record a command's side effects for ``pmt schema``.

    Apply it above ``@click.command`` so that it receives the built command.

    Parameters
    ----------
    effects : Iterable[str]
        Names from :data:`EFFECTS`. ``read_only`` excludes every writing effect.

    Returns
    -------
    Callable[[click.Command], click.Command]
        Decorator that stores the sorted effects on the command and returns it.

    Raises
    ------
    ValueError
        If an effect is unknown, none is given, or ``read_only`` is combined with
        a writing effect.
    """
    declared = frozenset(effects)
    unknown = declared - EFFECTS.keys()
    if unknown:
        raise ValueError(f'Unknown command effects: {", ".join(sorted(unknown))}')
    if not declared:
        raise ValueError('A command must declare at least one effect.')
    if 'read_only' in declared and declared & _WRITING_EFFECTS:
        raise ValueError('read_only cannot be combined with a writing effect.')

    def decorate(command: click.Command) -> click.Command:
        """Attach the validated effects to ``command``."""
        command.pmt_effects = tuple(sorted(declared))  # type: ignore[attr-defined]
        return command

    return decorate


def command_effects(command: click.Command) -> list[str]:
    """Return the effects declared for ``command``, or an empty list.

    Parameters
    ----------
    command : click.Command
        Command to inspect. Groups declare no effects of their own.

    Returns
    -------
    list[str]
        Sorted effect names.
    """
    return list(getattr(command, 'pmt_effects', ()))


def _json_default(value: Any) -> Any:
    """Convert values that workflow summaries hold but JSON cannot encode.

    NumPy scalars become Python numbers, paths become strings, and sets
    become sorted lists; anything else falls back to its string form.
    """
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    item = getattr(value, 'item', None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    tolist = getattr(value, 'tolist', None)
    if callable(tolist):
        return tolist()
    return str(value)


def emit_json(payload: Any) -> None:
    """Write one command result to stdout as a single JSON value.

    Parameters
    ----------
    payload : Any
        JSON-compatible result. NumPy scalars and arrays, paths, and sets
        are converted rather than rejected.
    """
    click.echo(json.dumps(payload, indent=2, default=_json_default))
