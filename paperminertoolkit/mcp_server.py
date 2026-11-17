"""Expose ``pmt`` commands as Model Context Protocol (MCP) tools over stdio.

``pmt mcp`` speaks the part of MCP a tool server needs: JSON-RPC 2.0
messages, one per line, on stdin and stdout, with the ``initialize``,
``ping``, ``tools/list`` and ``tools/call`` requests and the
``notifications/initialized`` notification. It needs no extra dependency.

Each tool is one JSON-capable leaf command. The tool name joins the command
path with underscores (``pmt corpus list`` becomes ``corpus_list``), the
input schema is generated from the Click parameters, and the declared side
effects become tool annotations. A tool call runs the command as a
subprocess (``python -m paperminertoolkit ... --json``) with stdin closed,
so workflow output can never corrupt the protocol stream, and returns the
command's JSON result, or its JSON error object, as the tool result.

By default only commands whose effects are ``read_only`` or
``migrates_corpus`` are listed; ``--allow`` admits more effects. Commands
that change settings, prompt, or start a server are never exposed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, TextIO

import click

from paperminertoolkit._agent import EFFECTS, command_effects, command_output_schema
from paperminertoolkit._version import __version__

#: Protocol versions this server answers to, newest first. Not yet tested
#: against every MCP client; a client asking for another version gets the newest.
PROTOCOL_VERSIONS = ('2025-06-18', '2025-03-26', '2024-11-05')

#: Effects every exposed tool may have without ``--allow``.
DEFAULT_EFFECTS = frozenset({'read_only', 'migrates_corpus'})

#: Effects that keep a command out of the server whatever ``--allow`` says.
NEVER_EXPOSED = frozenset({'writes_config', 'prompts', 'starts_server'})

#: Effects that ``--allow`` can add.
ALLOWABLE_EFFECTS = tuple(sorted(set(EFFECTS) - DEFAULT_EFFECTS - NEVER_EXPOSED))

_PARSE_ERROR, _INVALID_REQUEST, _METHOD_NOT_FOUND, _INVALID_PARAMS = -32700, -32600, -32601, -32602


def _json_flag(command: click.Command) -> bool:
    """Whether the command declares the shared ``--json`` flag."""
    return any('--json' in getattr(parameter, 'opts', ()) for parameter in command.params)


def _leaves(command: click.Command, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], click.Command]]:
    """Yield every runnable command below ``command`` with its path after ``pmt``."""
    if not isinstance(command, click.Group):
        yield path, command
        return
    context = click.Context(command)
    for name in command.list_commands(context):
        child = command.get_command(context, name)
        if child is not None:
            yield from _leaves(child, (*path, name))


def _type_schema(parameter: click.Parameter) -> dict[str, Any]:
    """Describe one parameter's value type in JSON Schema."""
    kind = parameter.type
    if isinstance(parameter, click.Option) and parameter.is_flag:
        item: dict[str, Any] = {'type': 'boolean'}
    elif isinstance(kind, click.Choice):
        item = {'type': 'string', 'enum': [str(choice) for choice in kind.choices]}
    elif isinstance(kind, (click.types.IntParamType, click.IntRange)):
        item = {'type': 'integer'}
        if isinstance(kind, click.IntRange):
            if kind.min is not None:
                item['minimum'] = kind.min
            if kind.max is not None:
                item['maximum'] = kind.max
    elif isinstance(kind, (click.types.FloatParamType, click.FloatRange)):
        item = {'type': 'number'}
    elif isinstance(kind, click.types.BoolParamType):
        item = {'type': 'boolean'}
    else:
        item = {'type': 'string'}
    many = getattr(parameter, 'multiple', False) or (
        isinstance(parameter, click.Argument) and parameter.nargs == -1)
    return {'type': 'array', 'items': item} if many else item


def _static_default(parameter: click.Parameter) -> tuple[bool, Any]:
    """Return whether a parameter has a plain JSON default, and that default.

    Click marks a missing default with a sentinel object and allows callable
    defaults; neither belongs in a schema or a command line.
    """
    default = parameter.default
    if isinstance(default, tuple):
        default = list(default)
    if isinstance(default, (str, int, float, bool)) or (
            isinstance(default, list) and all(isinstance(item, (str, int, float, bool))
                                              for item in default)):
        return True, default
    return False, None


def _input_schema(command: click.Command) -> dict[str, Any]:
    """Build a tool input schema from a command's Click parameters."""
    properties: dict[str, Any] = {}
    required = []
    for parameter in command.params:
        if parameter.name in {'json_output', 'help'} or not parameter.expose_value:
            continue
        schema = _type_schema(parameter)
        help_text = getattr(parameter, 'help', None)
        if help_text:
            schema['description'] = help_text
        has_default, default = _static_default(parameter)
        if has_default:
            schema['default'] = default
        properties[parameter.name] = schema
        if parameter.required:
            required.append(parameter.name)
    return {'type': 'object', 'properties': properties, 'required': required,
            'additionalProperties': False}


def exposed_commands(root: click.Group,
                     allow: Iterable[str] = ()) -> dict[str, tuple[tuple[str, ...], click.Command]]:
    """Select the commands the server offers as tools, keyed by tool name.

    Parameters
    ----------
    root : click.Group
        The ``pmt`` command group.
    allow : Iterable[str], optional
        Effects to admit beyond ``read_only`` and ``migrates_corpus``.

    Returns
    -------
    dict[str, tuple[tuple[str, ...], click.Command]]
        Tool name mapped to the command path and command.

    Raises
    ------
    ValueError
        If ``allow`` names an effect that cannot be allowed.
    """
    extra = set(allow)
    unknown = extra - set(ALLOWABLE_EFFECTS)
    if unknown:
        raise ValueError(f'Cannot allow: {", ".join(sorted(unknown))}. '
                         f'Allowable effects: {", ".join(ALLOWABLE_EFFECTS)}.')
    permitted = DEFAULT_EFFECTS | extra
    tools = {}
    for path, command in _leaves(root):
        effects = set(command_effects(command))
        if not effects or effects & NEVER_EXPOSED or not effects <= permitted:
            continue
        if not _json_flag(command) and command_output_schema(command) is None:
            continue
        tools['_'.join(path)] = (path, command)
    return tools


def tool_definition(name: str, path: tuple[str, ...], command: click.Command,
                    protocol_version: str) -> dict[str, Any]:
    """Describe one command as an MCP tool.

    Parameters
    ----------
    name : str
        Tool name.
    path : tuple[str, ...]
        Command path after ``pmt``.
    command : click.Command
        The command.
    protocol_version : str
        Negotiated protocol version; output schemas are sent from 2025-06-18.

    Returns
    -------
    dict[str, Any]
        The tool's ``name``, ``title``, ``description``, ``inputSchema``,
        ``annotations``, and, where the result is an object, ``outputSchema``.
    """
    effects = set(command_effects(command))
    definition: dict[str, Any] = {
        'name': name,
        'title': 'pmt ' + ' '.join(path),
        'description': (command.help or '').strip() + f'\n\nEffects: {", ".join(sorted(effects))}.',
        'inputSchema': _input_schema(command),
        'annotations': {
            'readOnlyHint': effects <= {'read_only'},
            'destructiveHint': bool(effects & {'writes_corpus', 'writes_files'}),
            'idempotentHint': effects <= {'read_only', 'migrates_corpus'},
            'openWorldHint': bool(effects & {'network', 'paid_model'}),
        },
    }
    output = command_output_schema(command)
    if protocol_version >= '2025-06-18' and output is not None and output.get('type') == 'object':
        definition['outputSchema'] = output
    return definition


def build_argv(path: tuple[str, ...], command: click.Command, arguments: Mapping[str, Any]) -> list[str]:
    """Turn tool arguments into the command line that runs the command.

    Parameters
    ----------
    path : tuple[str, ...]
        Command path after ``pmt``.
    command : click.Command
        The command, whose parameters decide each argument's form.
    arguments : Mapping[str, Any]
        Tool arguments keyed by parameter name.

    Returns
    -------
    list[str]
        Arguments after ``pmt``, ending with ``--json`` when the command has it.

    Raises
    ------
    ValueError
        If an argument is unknown or a required one is missing.
    """
    known = {parameter.name for parameter in command.params}
    unknown = set(arguments) - known - {'json_output'}
    if unknown:
        raise ValueError(f'Unknown arguments: {", ".join(sorted(unknown))}')
    # Positional values must keep their slots: an omitted optional argument
    # before a given one is filled with its default.
    slots = [parameter for parameter in command.params if isinstance(parameter, click.Argument)]
    last_given = max((index for index, parameter in enumerate(slots)
                      if parameter.name in arguments), default=-1)
    positional: list[str] = []
    for index, parameter in enumerate(slots[:last_given + 1]):
        has_default, default = _static_default(parameter)
        if parameter.name in arguments:
            value = arguments[parameter.name]
        elif has_default:
            value = default
        else:
            raise ValueError(f'Give {parameter.name} as well: a later positional argument is set.')
        positional.extend(str(item) for item in (value if isinstance(value, (list, tuple)) else [value]))
    for parameter in slots[last_given + 1:]:
        if parameter.required:
            raise ValueError(f'Missing required argument: {parameter.name}')
    options: list[str] = []
    for parameter in command.params:
        if isinstance(parameter, click.Argument) or parameter.name == 'json_output':
            continue
        if parameter.name not in arguments:
            if parameter.required:
                raise ValueError(f'Missing required argument: {parameter.name}')
            continue
        value = arguments[parameter.name]
        values = list(value) if isinstance(value, (list, tuple)) else [value]
        assert isinstance(parameter, click.Option)
        if parameter.is_flag:
            if value:
                options.append(parameter.opts[0])
            elif parameter.secondary_opts:
                options.append(parameter.secondary_opts[0])
            continue
        for item in values:
            options.extend([parameter.opts[0], str(item)])
    argv = [*path, *options, '--', *positional] if positional else [*path, *options]
    if _json_flag(command):
        argv.insert(len(path), '--json')
    return argv


class Server:
    """State of one MCP session: the exposed tools and the negotiated version.

    Parameters
    ----------
    root : click.Group
        The ``pmt`` command group.
    allow : Iterable[str], optional
        Effects to admit beyond the read-only defaults.
    timeout : float, optional
        Seconds a tool call may run before it is stopped.
    python : str, optional
        Interpreter that runs ``-m paperminertoolkit``; defaults to this one.
    """

    def __init__(self, root: click.Group, allow: Iterable[str] = (), timeout: float = 3600.0,
                 python: str | None = None) -> None:
        """Select the tools and remember how to run them."""
        self.tools = exposed_commands(root, allow)
        self.timeout = timeout
        self.python = python or sys.executable
        self.protocol_version = PROTOCOL_VERSIONS[0]

    def _call(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Run one tool as a subprocess and package its JSON as a tool result."""
        path, command = self.tools[name]
        argv = build_argv(path, command, arguments)
        try:
            completed = subprocess.run(
                [self.python, '-m', 'paperminertoolkit', *argv], stdin=subprocess.DEVNULL,
                capture_output=True, text=True, encoding='utf-8', timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            text = json.dumps({'error': {'type': 'operational', 'exit_code': 1, 'hint': None,
                                         'message': f'{name} exceeded {self.timeout:g} s'}})
            return {'content': [{'type': 'text', 'text': text}], 'isError': True}
        stdout = completed.stdout.strip()
        result: dict[str, Any] = {'isError': completed.returncode != 0}
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            payload = None
        if payload is None:
            stderr = completed.stderr.strip().splitlines()[-20:]
            text = stdout or '\n'.join(stderr) or f'exit {completed.returncode} without output'
        else:
            text = json.dumps(payload, indent=2)
            if isinstance(payload, dict) and self.protocol_version >= '2025-06-18':
                result['structuredContent'] = payload
        result['content'] = [{'type': 'text', 'text': text}]
        return result

    def handle(self, message: Any) -> dict[str, Any] | None:
        """Answer one JSON-RPC message; notifications get no answer.

        Parameters
        ----------
        message : Any
            Decoded JSON-RPC message.

        Returns
        -------
        dict[str, Any] or None
            The response, or ``None`` for a notification.
        """
        if not isinstance(message, dict) or message.get('jsonrpc') != '2.0' or 'method' not in message:
            return _error(message.get('id') if isinstance(message, dict) else None,
                          _INVALID_REQUEST, 'Invalid JSON-RPC request')
        method, params = message['method'], message.get('params') or {}
        if 'id' not in message:
            return None
        request_id = message['id']
        if method == 'initialize':
            requested = params.get('protocolVersion')
            self.protocol_version = requested if requested in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
            return _result(request_id, {
                'protocolVersion': self.protocol_version,
                'capabilities': {'tools': {'listChanged': False}},
                'serverInfo': {'name': 'paperminertoolkit', 'version': __version__},
                'instructions': ('Each tool runs one pmt command. Pass corpus and output paths '
                                 'explicitly; results are the command\'s JSON output.'),
            })
        if method == 'ping':
            return _result(request_id, {})
        if method == 'tools/list':
            return _result(request_id, {'tools': [
                tool_definition(name, path, command, self.protocol_version)
                for name, (path, command) in self.tools.items()]})
        if method == 'tools/call':
            name = params.get('name')
            if name not in self.tools:
                return _error(request_id, _INVALID_PARAMS, f'Unknown or disallowed tool: {name}')
            try:
                return _result(request_id, self._call(name, params.get('arguments') or {}))
            except ValueError as error:
                return _error(request_id, _INVALID_PARAMS, str(error))
        return _error(request_id, _METHOD_NOT_FOUND, f'Method not found: {method}')


def _result(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    """Build a JSON-RPC success response."""
    return {'jsonrpc': '2.0', 'id': request_id, 'result': result}


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    """Build a JSON-RPC error response."""
    return {'jsonrpc': '2.0', 'id': request_id, 'error': {'code': code, 'message': message}}


def serve(server: Server, stdin: TextIO, stdout: TextIO) -> None:
    """Answer newline-delimited JSON-RPC messages until stdin closes.

    Parameters
    ----------
    server : Server
        Session state and tools.
    stdin : TextIO
        Stream of incoming messages, one JSON object per line.
    stdout : TextIO
        Stream for responses; nothing else is written to it.
    """
    for line in stdin:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as error:
            response: dict[str, Any] | None = _error(None, _PARSE_ERROR, f'Parse error: {error}')
        else:
            response = server.handle(message)
        if response is not None:
            stdout.write(json.dumps(response) + '\n')
            stdout.flush()


def client_config(allow: Iterable[str] = ()) -> dict[str, Any]:
    """Return an ``mcpServers`` entry that starts this server with this interpreter.

    Parameters
    ----------
    allow : Iterable[str], optional
        Effects to pass with ``--allow``.

    Returns
    -------
    dict[str, Any]
        ``{"mcpServers": {"paperminertoolkit": {"command", "args"}}}``.
    """
    args = ['-m', 'paperminertoolkit', 'mcp']
    for effect in allow:
        args.extend(['--allow', effect])
    return {'mcpServers': {'paperminertoolkit': {'command': str(Path(sys.executable)),
                                                 'args': args}}}
