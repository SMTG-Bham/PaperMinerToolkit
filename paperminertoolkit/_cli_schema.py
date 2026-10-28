"""Describe the live command tree without executing workflow callbacks."""

from __future__ import annotations

import json
from typing import Any

import click

from paperminertoolkit._version import __version__


def _json_default(value: Any) -> str | list[Any]:
    """Represent unusual static defaults without evaluating callable defaults."""
    if callable(value):
        return f'<callable {getattr(value, "__name__", type(value).__name__)}>'
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    return str(value)


def _command_schema(command: click.Command, context: click.Context,
                    path: tuple[str, ...]) -> dict[str, Any]:
    """Read command and parameter declarations without parsing their values."""
    declared = command.get_params(context)
    # Click 8.5 stores the implicit help option under the reserved name
    # ``_click_default_help`` so that a command may declare its own ``help``
    # parameter. The schema keeps calling it ``help``, so its output does not
    # change with the installed Click, unless the command has claimed that name.
    help_option = command.get_help_option(context)
    help_name_free = all(parameter.name != 'help' for parameter in declared
                         if parameter is not help_option)
    parameters = []
    for parameter in declared:
        info = parameter.to_info_dict()
        if parameter is help_option and help_name_free:
            info['name'] = 'help'
        if isinstance(parameter, click.Option):
            info.update(
                prompt_required=parameter.prompt_required,
                hide_input=parameter.hide_input,
                confirmation_prompt=parameter.confirmation_prompt,
            )
        parameters.append(info)

    children = []
    if isinstance(command, click.Group):
        for name in command.list_commands(context):
            child = command.get_command(context, name)
            if child is not None:
                child_context = child.context_class(
                    child, info_name=name, parent=context, **child.context_settings,
                )
                children.append(_command_schema(child, child_context, (*path, name)))

    return {
        'name': path[-1],
        'path': ' '.join(path),
        'help': command.help,
        'parameters': parameters,
        'commands': children,
    }


@click.command('schema')
@click.argument('command_path', nargs=-1)
@click.pass_context
def schema_command(context: click.Context, command_path: tuple[str, ...]) -> None:
    """Print the command tree as JSON, optionally scoped to COMMAND_PATH.

    For example, `pmt schema config model` describes model configuration.
    Defaults and prompts describe declarations only; configured values and
    environment variable contents are never included.
    """
    selected_context = context.find_root()
    selected_command = selected_context.command
    path = ('pmt',)
    for name in command_path:
        child = (selected_command.get_command(selected_context, name)
                 if isinstance(selected_command, click.Group) else None)
        if child is None:
            raise click.UsageError(
                f'Unknown command path: {" ".join((*path, name))}', ctx=context,
            )
        selected_context = child.context_class(
            child, info_name=name, parent=selected_context, **child.context_settings,
        )
        selected_command = child
        path = (*path, name)

    result = {
        'schema_version': 1,
        'package_version': __version__,
        'command': _command_schema(selected_command, selected_context, path),
    }
    click.echo(json.dumps(result, indent=2, default=_json_default))
