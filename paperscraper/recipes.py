"""Load and validate extraction recipes.

Recipes describe the material type, target fields, examples, aliases, and unit
expectations used by scraping and storage. This module validates recipe shape
and maps extracted columns back to canonical output columns.
"""

import json
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
RECIPES_PATH = MODULE_DIR / 'resources' / 'recipes.json'
METADATA_FIELDS = ['Paper id', 'doi', 'Publication date', 'Source', 'Source path']


def _validate_recipe(recipe, source: str):
    """Validate the minimum recipe structure.

    Parameters
    ----------
    recipe : dict
        Recipe mapping to validate.
    source : str
        Human-readable source used in validation errors.

    Returns
    -------
    dict
        The validated recipe with optional defaults populated.

    Raises
    ------
    ValueError
        If the recipe is not a mapping, lacks required fields, or has no valid
        search-field mapping.
    """
    if not isinstance(recipe, dict):
        raise ValueError(f'Recipe in {source} must be a JSON object.')
    missing = [key for key in ['material type', 'search fields'] if key not in recipe]
    if missing:
        raise ValueError(f'Recipe in {source} is missing required key(s): {", ".join(missing)}.')
    if not isinstance(recipe['search fields'], dict) or not recipe['search fields']:
        raise ValueError(f'Recipe in {source} must define one or more search fields.')
    recipe.setdefault('additional prompts', '')
    return recipe


def _load_recipe_file(path: Path):
    """Load a recipe from a standalone JSON file.

    Parameters
    ----------
    path : pathlib.Path
        Recipe JSON file to read.

    Returns
    -------
    dict
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

    if isinstance(data, dict) and 'material type' in data and 'search fields' in data:
        return _validate_recipe(data, str(path))
    if isinstance(data, dict) and len(data) == 1:
        recipe_name, recipe = next(iter(data.items()))
        return _validate_recipe(recipe, f'{path}:{recipe_name}')
    raise ValueError(
        f'Recipe file {path} must contain a single recipe object, or a JSON object with exactly one named recipe.'
    )


def load_recipe(recipe_name: str):
    """Load a bundled recipe or a standalone recipe file.

    Parameters
    ----------
    recipe_name : str
        Bundled recipe name or path to a recipe JSON file.

    Returns
    -------
    dict
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
        raise FileNotFoundError('The recipes.json file is missing. Please reinstall PaperScraper.')
    except json.JSONDecodeError as e:
        raise ValueError('The recipes.json file is not valid JSON. Please reinstall PaperScraper.') from e

    try:
        return _validate_recipe(recipes[recipe_name.lower()], recipe_name)
    except KeyError as e:
        raise KeyError(
            f'Recipe called "{recipe_name}" does not exist, and no recipe file was found at that path.') from e


def field_columns(recipe, existing_columns=None):
    """Build output columns for a recipe.

    Parameters
    ----------
    recipe : dict
        Validated extraction recipe.
    existing_columns : iterable of str, optional
        Existing output columns to preserve instead of deriving new ones.

    Returns
    -------
    list of str
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


def aliases_for(recipe):
    """Build aliases for recipe and metadata fields.

    Parameters
    ----------
    recipe : dict
        Validated extraction recipe.

    Returns
    -------
    dict of str to set of str
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


def canonical_match(series_name, columns, recipe):
    """Match an incoming name to a canonical output column.

    Parameters
    ----------
    series_name : str
        Incoming scrape-result column name.
    columns : iterable of str
        Canonical output columns available for matching.
    recipe : dict
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
