"""Verify machine-readable CLI discovery and module entry points."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, NoReturn

import click
import pytest
from click.testing import CliRunner

import paperminertoolkit.cli as cli
import paperminertoolkit.settings as settings
from paperminertoolkit._cli_schema import schema_command
from paperminertoolkit._version import __version__


def _assert_live_tree(command: click.Command, node: dict[str, Any],
                      path: tuple[str, ...]) -> None:
    """Compare every described command and parameter against its declaration."""
    context = click.Context(command)
    assert node['name'] == path[-1]
    assert node['path'] == ' '.join(path)
    assert node['help'] == command.help
    assert [parameter['name'] for parameter in node['parameters']] == [
        parameter.name for parameter in command.get_params(context)
    ]
    if isinstance(command, click.Group):
        assert [child['name'] for child in node['commands']] == command.list_commands(context)
        for child in node['commands']:
            child_command = command.get_command(context, child['name'])
            assert child_command is not None
            _assert_live_tree(child_command, child, (*path, child['name']))
    else:
        assert node['commands'] == []


def test_schema_matches_the_complete_live_cli() -> None:
    """Include every current command and implicit help and version options."""
    result = CliRunner().invoke(cli.main, ['schema'])

    assert result.exit_code == 0, result.output
    assert result.stderr == ''
    payload = json.loads(result.stdout)
    assert payload['schema_version'] == 1
    assert payload['package_version'] == __version__
    _assert_live_tree(cli.main, payload['command'], ('pmt',))
    assert {'help', 'version'} <= {
        parameter['name'] for parameter in payload['command']['parameters']
    }


def test_schema_reports_choices_defaults_repeatability_and_prompts() -> None:
    """Expose enough parameter metadata to build valid noninteractive calls."""
    search_result = CliRunner().invoke(cli.main, ['schema', 'search'])
    assert search_result.exit_code == 0, search_result.output
    search = json.loads(search_result.stdout)['command']
    parameters = {parameter['name']: parameter for parameter in search['parameters']}

    assert search['path'] == 'pmt search'
    assert search['commands'] == []
    assert parameters['source']['multiple'] is True
    assert parameters['source']['default'] == ['all']
    assert 'all' in parameters['source']['type']['choices']
    assert parameters['source']['required'] is False
    assert parameters['count']['default'] == 200
    assert parameters['count']['type']['min'] == 1
    assert parameters['db_path']['param_type_name'] == 'argument'
    assert parameters['help']['opts'] == ['--help']

    model_result = CliRunner().invoke(cli.main, ['schema', 'config', 'model'])
    assert model_result.exit_code == 0, model_result.output
    model = json.loads(model_result.stdout)['command']
    assert model['path'] == 'pmt config model'
    model_parameters = {parameter['name']: parameter for parameter in model['parameters']}
    assert model_parameters['provider']['prompt']
    assert model_parameters['provider']['prompt_required'] is True
    assert model_parameters['provider']['hide_input'] is False


@pytest.mark.parametrize('path', [['missing'], ['config', 'missing'], ['search', 'missing']])
def test_unknown_schema_path_is_a_usage_error(path: list[str]) -> None:
    """Reject invalid paths on stderr before producing a JSON payload."""
    result = CliRunner().invoke(cli.main, ['schema', *path])

    assert result.exit_code == 2
    assert result.stdout == ''
    assert 'Unknown command path:' in result.stderr


def test_schema_does_not_execute_callbacks_defaults_or_read_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Describe dynamic defaults and secret environment bindings without reading them."""
    def forbidden(*args: Any, **kwargs: Any) -> NoReturn:
        """Fail if discovery calls any settings or workflow implementation."""
        raise AssertionError('Discovery must not invoke this callable.')

    @click.command('dynamic')
    @click.option('--token', envvar='PMT_SCHEMA_TEST_SECRET', default=forbidden)
    @click.option('--output', default=Path('output.json'), type=click.Path())
    def dynamic(token: str, output: str) -> None:
        """Exercise a command that must not run during discovery."""
        forbidden()

    monkeypatch.setattr(settings, 'load_settings', forbidden)
    monkeypatch.setenv('PMT_SCHEMA_TEST_SECRET', 'secret-value-not-for-discovery')
    monkeypatch.setitem(cli.main.commands, 'dynamic', dynamic)
    result = CliRunner().invoke(cli.main, ['schema', 'dynamic'])

    assert result.exit_code == 0, result.output
    assert 'secret-value-not-for-discovery' not in result.stdout
    parameters = {parameter['name']: parameter for parameter in
                  json.loads(result.stdout)['command']['parameters']}
    assert parameters['token']['default'] == '<callable forbidden>'
    assert parameters['token']['envvar'] == 'PMT_SCHEMA_TEST_SECRET'
    assert parameters['output']['default'] == 'output.json'


def test_schema_never_parses_child_defaults_or_invokes_parameter_callbacks() -> None:
    """Read declarations even for required or prompted options with callbacks."""
    def forbidden(*args: Any, **kwargs: Any) -> NoReturn:
        """Reject any parameter conversion or callback execution."""
        raise AssertionError('Parameter callback was executed.')

    root = click.Group('temporary')
    root.add_command(schema_command)
    child = click.Command('child', callback=forbidden, context_settings={
        'help_option_names': ['-h', '--help'],
    }, params=[
        click.Option(['--secret'], required=True, prompt=True, hide_input=True,
                     confirmation_prompt=True, callback=forbidden),
    ])
    root.add_command(child)

    result = CliRunner().invoke(root, ['schema'])

    assert result.exit_code == 0, result.output
    node = next(child for child in json.loads(result.stdout)['command']['commands']
                if child['name'] == 'child')
    secret = next(parameter for parameter in node['parameters'] if parameter['name'] == 'secret')
    assert secret['required'] is True
    assert secret['hide_input'] is True
    assert secret['confirmation_prompt'] is True
    help_option = next(parameter for parameter in node['parameters'] if parameter['name'] == 'help')
    assert set(help_option['opts']) == {'-h', '--help'}


@pytest.mark.parametrize('option', ['--help', '--version'])
def test_module_invocation_supports_help_and_version(option: str) -> None:
    """Allow agents to invoke the CLI using an explicitly selected interpreter."""
    result = subprocess.run(
        [sys.executable, '-m', 'paperminertoolkit', option],
        cwd=Path(__file__).parents[1],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr == ''
    assert ('schema' if option == '--help' else __version__) in result.stdout
