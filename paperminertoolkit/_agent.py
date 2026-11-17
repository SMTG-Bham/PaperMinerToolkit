"""Machine-readable command metadata for agents that drive the ``pmt`` CLI.

Each leaf command declares its side effects with :func:`agent_meta`. The
``pmt schema`` output carries them, so a caller can tell which commands make
network requests, call a language model, or change the corpus before running
them.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Mapping
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


def agent_meta(*, effects: Iterable[str],
               output: Mapping[str, Any] | None = None) -> Callable[[click.Command], click.Command]:
    """Record a command's side effects and JSON output shape for ``pmt schema``.

    Apply it above ``@click.command`` so that it receives the built command.

    Parameters
    ----------
    effects : Iterable[str]
        Names from :data:`EFFECTS`. ``read_only`` excludes every writing effect.
    output : Mapping[str, Any] or None, optional
        JSON Schema of the command's JSON result, using only the keywords
        :func:`validate_json` supports.

    Returns
    -------
    Callable[[click.Command], click.Command]
        Decorator that stores the sorted effects on the command and returns it.

    Raises
    ------
    ValueError
        If an effect is unknown, none is given, ``read_only`` is combined with
        a writing effect, or the output schema uses an unsupported keyword.
    """
    declared = frozenset(effects)
    unknown = declared - EFFECTS.keys()
    if unknown:
        raise ValueError(f'Unknown command effects: {", ".join(sorted(unknown))}')
    if not declared:
        raise ValueError('A command must declare at least one effect.')
    if 'read_only' in declared and declared & _WRITING_EFFECTS:
        raise ValueError('read_only cannot be combined with a writing effect.')
    if output is not None:
        check_schema(output)

    def decorate(command: click.Command) -> click.Command:
        """Attach the validated effects and output schema to ``command``."""
        command.pmt_effects = tuple(sorted(declared))  # type: ignore[attr-defined]
        command.pmt_output = output  # type: ignore[attr-defined]
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


def command_output_schema(command: click.Command) -> Mapping[str, Any] | None:
    """Return the JSON Schema declared for ``command``'s JSON result, if any.

    Parameters
    ----------
    command : click.Command
        Command to inspect.

    Returns
    -------
    Mapping[str, Any] or None
        The declared schema, or ``None`` for commands without JSON output.
    """
    return getattr(command, 'pmt_output', None)


#: JSON Schema keywords :func:`validate_json` understands. Others are rejected
#: when a schema is declared, so a schema can never promise more than is checked.
SCHEMA_KEYWORDS = frozenset({'type', 'properties', 'required', 'items', 'additionalProperties',
                             'enum', 'anyOf', 'description', 'title', '$schema'})
_JSON_TYPES = {
    'object': lambda value: isinstance(value, dict),
    'array': lambda value: isinstance(value, list),
    'string': lambda value: isinstance(value, str),
    'boolean': lambda value: isinstance(value, bool),
    'integer': lambda value: isinstance(value, int) and not isinstance(value, bool),
    'number': lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
    'null': lambda value: value is None,
}


def check_schema(schema: Mapping[str, Any], path: str = '#') -> None:
    """Reject schemas that use keywords :func:`validate_json` would ignore.

    Parameters
    ----------
    schema : Mapping[str, Any]
        JSON Schema to check, recursively.
    path : str, optional
        Location used in error messages.

    Raises
    ------
    ValueError
        If a keyword or type name is unsupported.
    """
    unknown = set(schema) - SCHEMA_KEYWORDS
    if unknown:
        raise ValueError(f'Unsupported JSON Schema keywords at {path}: {", ".join(sorted(unknown))}')
    types = schema.get('type', [])
    for name in [types] if isinstance(types, str) else types:
        if name not in _JSON_TYPES:
            raise ValueError(f'Unsupported JSON type at {path}: {name}')
    for key, child in schema.get('properties', {}).items():
        check_schema(child, f'{path}/properties/{key}')
    for key in ('items', 'additionalProperties'):
        if isinstance(schema.get(key), Mapping):
            check_schema(schema[key], f'{path}/{key}')
    for index, child in enumerate(schema.get('anyOf', [])):
        check_schema(child, f'{path}/anyOf/{index}')


def validate_json(instance: Any, schema: Mapping[str, Any], path: str = '$') -> list[str]:
    """Check a decoded JSON value against a schema in the supported subset.

    Parameters
    ----------
    instance : Any
        Value decoded from JSON.
    schema : Mapping[str, Any]
        Schema using :data:`SCHEMA_KEYWORDS`.
    path : str, optional
        Location of ``instance``, used in messages.

    Returns
    -------
    list[str]
        One message per violation; empty when the value conforms.
    """
    if 'anyOf' in schema:
        branches = [validate_json(instance, branch, path) for branch in schema['anyOf']]
        if all(branches):
            best = min(branches, key=len)
            return [f'{path}: matches no allowed shape; closest: {best[0]}'] if best else []
    types = schema.get('type')
    if types is not None:
        names = [types] if isinstance(types, str) else list(types)
        if not any(_JSON_TYPES[name](instance) for name in names):
            return [f'{path}: expected {" or ".join(names)}, got {type(instance).__name__}']
    errors = []
    if 'enum' in schema and instance not in schema['enum']:
        errors.append(f'{path}: {instance!r} is not one of {schema["enum"]!r}')
    if isinstance(instance, dict):
        for key in schema.get('required', []):
            if key not in instance:
                errors.append(f'{path}: missing required key {key!r}')
        properties = schema.get('properties', {})
        extra = schema.get('additionalProperties', True)
        for key, value in instance.items():
            if key in properties:
                errors.extend(validate_json(value, properties[key], f'{path}.{key}'))
            elif extra is False:
                errors.append(f'{path}: unexpected key {key!r}')
            elif isinstance(extra, Mapping):
                errors.extend(validate_json(value, extra, f'{path}.{key}'))
    if isinstance(instance, list) and isinstance(schema.get('items'), Mapping):
        for index, item in enumerate(instance):
            errors.extend(validate_json(item, schema['items'], f'{path}[{index}]'))
    return errors


#: Set by the test suite: emit_json then checks every result against its schema.
VALIDATE_OUTPUT_ENV = 'PAPERMINERTOOLKIT_VALIDATE_OUTPUT'


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

    Raises
    ------
    AssertionError
        If :data:`VALIDATE_OUTPUT_ENV` is set and the result does not match
        the running command's declared output schema.
    """
    text = json.dumps(payload, indent=2, default=_json_default)
    context = click.get_current_context(silent=True)
    if context is not None:
        context.find_root().meta['pmt_json_emitted'] = True
        schema = command_output_schema(context.command)
        if os.environ.get(VALIDATE_OUTPUT_ENV) and schema is not None:
            errors = validate_json(json.loads(text), schema)
            if errors:
                raise AssertionError(f'{context.command_path} JSON does not match its output '
                                     f'schema: {"; ".join(errors[:5])}')
    click.echo(text)
