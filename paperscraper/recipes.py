"""Load and validate extraction recipes.

Recipes define the records to extract, target fields, examples, aliases, and
unit expectations used by scraping and storage. This module validates recipe
shape and maps extracted columns back to canonical output columns.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypeAlias

MODULE_DIR = Path(__file__).resolve().parent
RECIPES_PATH = MODULE_DIR / 'resources' / 'recipes.json'
METADATA_FIELDS = ['Paper id', 'doi', 'Publication date', 'Source', 'Source path']
_Recipe: TypeAlias = dict[str, Any]


def _validate_recipe(recipe: object, source: str) -> _Recipe:
    """Validate the minimum recipe structure.

    Parameters
    ----------
    recipe : object
        Recipe mapping to validate.
    source : str
        Human-readable source used in validation errors.

    Returns
    -------
    _Recipe
        The validated recipe with optional defaults populated.

    Raises
    ------
    ValueError
        If the recipe is not a mapping, lacks required fields, or has no valid
        search-field mapping.
    """
    if not isinstance(recipe, dict):
        raise ValueError(f'Recipe in {source} must be a JSON object.')
    missing = [key for key in ['record definition', 'search fields'] if key not in recipe]
    if missing:
        raise ValueError(f'Recipe in {source} is missing required key(s): {", ".join(missing)}.')
    if not isinstance(recipe['search fields'], dict) or not recipe['search fields']:
        raise ValueError(f'Recipe in {source} must define one or more search fields.')

    definition = recipe['record definition']
    if not isinstance(definition, dict):
        raise ValueError(f'Recipe in {source} must define "record definition" as a JSON object.')
    definition_keys = ['subject', 'singular', 'plural', 'unit', 'identity fields']
    missing_definition = [key for key in definition_keys if key not in definition]
    if missing_definition:
        raise ValueError(
            f'Recipe in {source} record definition is missing required key(s): '
            f'{", ".join(missing_definition)}.'
        )
    for key in ['subject', 'singular', 'plural', 'unit']:
        if not isinstance(definition[key], str) or not definition[key].strip():
            raise ValueError(f'Recipe in {source} record definition "{key}" must be a non-empty string.')
    identity_fields = definition['identity fields']
    if not isinstance(identity_fields, list) or any(
        not isinstance(field, str) or not field.strip() for field in identity_fields
    ):
        raise ValueError(f'Recipe in {source} record definition "identity fields" must be a list of strings.')
    unknown_identity_fields = [field for field in identity_fields if field not in recipe['search fields']]
    if unknown_identity_fields:
        raise ValueError(
            f'Recipe in {source} record definition references unknown identity field(s): '
            f'{", ".join(unknown_identity_fields)}.'
        )
    recipe.setdefault('additional prompts', '')
    return recipe


def _load_recipe_file(path: Path) -> _Recipe:
    """Load a recipe from a standalone JSON file.

    Parameters
    ----------
    path : pathlib.Path
        Recipe JSON file to read.

    Returns
    -------
    _Recipe
        Validated recipe data.

    Raises
    ------
    ValueError
        If the file contains invalid JSON, an ambiguous structure, or an
        invalid recipe.
    OSError
        If the recipe file cannot be read.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f'Recipe file {path} is not valid JSON: {e}') from e

    if isinstance(data, dict) and 'record definition' in data and 'search fields' in data:
        return _validate_recipe(data, str(path))
    if isinstance(data, dict) and len(data) == 1:
        recipe_name, recipe = next(iter(data.items()))
        return _validate_recipe(recipe, f'{path}:{recipe_name}')
    raise ValueError(
        f'Recipe file {path} must contain a single recipe object, or a JSON object with exactly one named recipe.'
    )


def load_recipe(recipe_name: str) -> _Recipe:
    """Load a bundled recipe or a standalone recipe file.

    Parameters
    ----------
    recipe_name : str
        Bundled recipe name or path to a recipe JSON file.

    Returns
    -------
    _Recipe
        Validated recipe data.

    Raises
    ------
    FileNotFoundError
        If the bundled recipe resource is missing.
    ValueError
        If recipe JSON or its recipe structure is invalid.
    KeyError
        If no bundled or standalone recipe matches ``recipe_name``.
    OSError
        If a recipe file cannot be read.
    """
    recipe_path = Path(recipe_name).expanduser()
    if recipe_path.is_file():
        return _load_recipe_file(recipe_path)

    try:
        with open(RECIPES_PATH, 'r', encoding='utf-8') as f:
            recipes = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError('The recipes.json file is missing. Please reinstall PaperMiner.')
    except json.JSONDecodeError as e:
        raise ValueError('The recipes.json file is not valid JSON. Please reinstall PaperMiner.') from e

    try:
        return _validate_recipe(recipes[recipe_name.lower()], recipe_name)
    except KeyError as e:
        raise KeyError(
            f'Recipe called "{recipe_name}" does not exist, and no recipe file was found at that path.') from e


def field_columns(recipe: _Recipe, existing_columns: Iterable[str] | None = None) -> list[str]:
    """Build output columns for a recipe.

    Parameters
    ----------
    recipe : _Recipe
        Validated extraction recipe.
    existing_columns : Iterable[str] or None, optional
        Existing output columns to preserve instead of deriving new ones.

    Returns
    -------
    list[str]
        Existing columns when provided; otherwise recipe fields with unit
        labels followed by metadata columns.
    """
    columns = list(existing_columns or [])
    if columns:
        return columns
    for field, config in recipe['search fields'].items():
        if 'unit' in config:
            columns.append(f'{field} [{config["unit"]}]')
        else:
            columns.append(field)
    columns.extend(METADATA_FIELDS)
    return columns


def aliases_for(recipe: _Recipe) -> dict[str, set[str]]:
    """Build aliases for recipe and metadata fields.

    Parameters
    ----------
    recipe : _Recipe
        Validated extraction recipe.

    Returns
    -------
    dict[str, set[str]]
        Canonical fields mapped to their lower-case aliases.
    """
    aliases = {}
    for field, config in recipe['search fields'].items():
        names = {field.lower()}
        prompt = config.get('prompt')
        if prompt:
            names.add(prompt.lower())
        for alias in config.get('aliases', []):
            names.add(alias.lower())
        aliases[field] = names
    for field in METADATA_FIELDS:
        aliases[field] = {field.lower()}
    return aliases


def canonical_match(series_name: str, columns: Iterable[str], recipe: _Recipe) -> str | None:
    """Match an incoming name to a canonical output column.

    Parameters
    ----------
    series_name : str
        Incoming scrape-result column name.
    columns : Iterable[str]
        Canonical output columns available for matching.
    recipe : _Recipe
        Validated extraction recipe containing aliases.

    Returns
    -------
    str or None
        Matching canonical column, or ``None`` when no match exists.
    """
    raw_name = series_name.strip()
    normalized = raw_name.lower()
    aliases = aliases_for(recipe)
    for column in columns:
        base = column.split(' [')[0]
        if normalized in aliases.get(base, {base.lower()}):
            return column
    for column in columns:
        base = column.split(' [')[0]
        if normalized == base.lower():
            return column
    return None
