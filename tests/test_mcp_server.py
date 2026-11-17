"""Exercise the stdio MCP server: tool selection, schemas, argv mapping, and sessions."""

from __future__ import annotations

import io
import json
import re
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

import paperminertoolkit.corpus.database as corpus
from paperminertoolkit import cli, mcp_server


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Create a corpus with two papers."""
    path = tmp_path / 'papers.db'
    with closing(corpus.connect(path)) as conn, conn:
        corpus.upsert_paper(conn, {'paper_id': 'paper:1', 'title': 'LLZO'})
        corpus.upsert_paper(conn, {'paper_id': 'paper:2', 'title': 'LATP'})
    return path


def _session(messages: list[Any], **server_options: Any) -> list[dict[str, Any]]:
    """Run the server over one sequence of messages and return its responses."""
    server = mcp_server.Server(cli.main, **server_options)
    stdin = io.StringIO(''.join((message if isinstance(message, str) else json.dumps(message)) + '\n'
                                for message in messages))
    stdout = io.StringIO()
    mcp_server.serve(server, stdin, stdout)
    return [json.loads(line) for line in stdout.getvalue().splitlines()]


def test_default_tools_are_read_only_and_well_named() -> None:
    """Expose only read-only JSON commands, never setters, servers, or prompting commands."""
    tools = mcp_server.exposed_commands(cli.main)
    assert {'corpus_list', 'corpus_show', 'schema', 'recipe_list', 'status',
            'scrape-progress', 'config_status'} <= set(tools)
    assert not {'scrape', 'download', 'store', 'mcp', 'validate_gui', 'config_model',
                'config_openai-key', 'corpus_export', 'probe'} & set(tools)
    assert all(re.fullmatch(r'[A-Za-z0-9_-]{1,64}', name) for name in tools)
    wider = mcp_server.exposed_commands(cli.main, allow=['network', 'writes_corpus', 'writes_files',
                                                          'paid_model'])
    assert {'scrape', 'download', 'probe', 'corpus_export'} <= set(wider)
    assert not {'store', 'mcp', 'config_model', 'config_openai-key'} & set(wider)
    with pytest.raises(ValueError, match='Cannot allow: prompts'):
        mcp_server.exposed_commands(cli.main, allow=['prompts'])


def test_tool_definitions_carry_input_output_schemas_and_annotations() -> None:
    """Describe parameters, result shape, and effects in each tool."""
    path, command = mcp_server.exposed_commands(cli.main)['corpus_list']
    tool = mcp_server.tool_definition('corpus_list', path, command, '2025-06-18')
    properties = tool['inputSchema']['properties']
    assert properties['db_path'] == {'type': 'string', 'default': 'papers.db'}
    assert properties['limit']['type'] == 'integer' and properties['limit']['maximum'] == 1000
    assert 'json_output' not in properties
    assert tool['outputSchema']['required'] == ['papers', 'total', 'limit', 'offset']
    assert tool['annotations'] == {'readOnlyHint': True, 'destructiveHint': False,
                                   'idempotentHint': True, 'openWorldHint': False}
    older = mcp_server.tool_definition('corpus_list', path, command, '2025-03-26')
    assert 'outputSchema' not in older


def test_build_argv_maps_flags_lists_and_positional_slots() -> None:
    """Keep positional slots, repeat list options, and use the secondary flag for false."""
    tools = mcp_server.exposed_commands(cli.main, allow=['network', 'writes_corpus'])
    path, command = tools['download']
    argv = mcp_server.build_argv(path, command, {
        'db_path': 'c.db', 'sources': ['arxiv', 'pubmed'], 'download_abstract': False,
        'force': True})
    assert argv == ['download', '--json', '--source', 'arxiv', '--source', 'pubmed',
                    '--no-abstract', '--force', '--', 'c.db']
    path, command = tools['search']
    assert mcp_server.build_argv(path, command, {'db_path': 'c.db'})[-2:] == [
        'Lithium solid electrolyte', 'c.db']
    with pytest.raises(ValueError, match='Unknown arguments: nonsense'):
        mcp_server.build_argv(path, command, {'nonsense': 1})
    path, command = mcp_server.exposed_commands(cli.main)['corpus_show']
    with pytest.raises(ValueError, match='Missing required argument'):
        mcp_server.build_argv(path, command, {})


def test_session_lists_and_calls_tools_through_subprocesses(db_path: Path) -> None:
    """Negotiate the version, ignore notifications, and return command JSON as results."""
    responses = _session([
        {'jsonrpc': '2.0', 'id': 1, 'method': 'initialize',
         'params': {'protocolVersion': '2025-06-18', 'capabilities': {},
                    'clientInfo': {'name': 'test', 'version': '0'}}},
        {'jsonrpc': '2.0', 'method': 'notifications/initialized'},
        {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list'},
        {'jsonrpc': '2.0', 'id': 3, 'method': 'tools/call',
         'params': {'name': 'corpus_list', 'arguments': {'db_path': str(db_path), 'limit': 1}}},
        {'jsonrpc': '2.0', 'id': 4, 'method': 'tools/call',
         'params': {'name': 'corpus_show', 'arguments': {'db_path': str(db_path),
                                                         'paper_id': 'paper:404'}}},
        {'jsonrpc': '2.0', 'id': 5, 'method': 'ping'},
    ])
    assert [response['id'] for response in responses] == [1, 2, 3, 4, 5]
    assert responses[0]['result']['protocolVersion'] == '2025-06-18'
    assert responses[0]['result']['capabilities'] == {'tools': {'listChanged': False}}
    assert 'corpus_list' in {tool['name'] for tool in responses[1]['result']['tools']}
    listed = responses[2]['result']
    assert listed['isError'] is False
    assert listed['structuredContent']['total'] == 2
    assert json.loads(listed['content'][0]['text']) == listed['structuredContent']
    failed = responses[3]['result']
    assert failed['isError'] is True
    assert failed['structuredContent']['error']['type'] == 'not_found'
    assert responses[4]['result'] == {}


def test_session_reports_protocol_errors() -> None:
    """Answer malformed, unknown, and disallowed requests with JSON-RPC errors."""
    responses = _session([
        'not json',
        {'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {'protocolVersion': '1999-01-01'}},
        {'jsonrpc': '2.0', 'id': 2, 'method': 'resources/list'},
        {'jsonrpc': '2.0', 'id': 3, 'method': 'tools/call', 'params': {'name': 'scrape'}},
        {'jsonrpc': '2.0', 'id': 4, 'method': 'tools/call',
         'params': {'name': 'corpus_list', 'arguments': {'bogus': True}}},
        {'id': 5, 'method': 'ping'},
    ])
    assert responses[0]['error']['code'] == -32700
    assert responses[1]['result']['protocolVersion'] == mcp_server.PROTOCOL_VERSIONS[0]
    assert responses[2]['error']['code'] == -32601
    assert responses[3]['error'] == {'code': -32602, 'message': 'Unknown or disallowed tool: scrape'}
    assert responses[4]['error']['code'] == -32602
    assert responses[5]['error']['code'] == -32600


def test_mcp_command_prints_a_client_config() -> None:
    """Give an mcpServers entry that starts the server with this interpreter."""
    result = CliRunner().invoke(cli.main, ['mcp', '--allow', 'network', '--print-config'])
    assert result.exit_code == 0, result.output
    entry = json.loads(result.stdout)['mcpServers']['paperminertoolkit']
    assert entry['args'] == ['-m', 'paperminertoolkit', 'mcp', '--allow', 'network']
