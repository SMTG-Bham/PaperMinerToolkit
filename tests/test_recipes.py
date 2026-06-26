"""Unit tests for paperscraper.recipes.

This module tests recipe validation, loading bundled and file-based recipes,
building output columns, alias collection, and canonical column matching.
"""

import importlib
import json

import pytest

recipes = importlib.import_module('paperscraper.recipes')


def sample_recipe():
    """Return a minimal valid recipe for recipe unit tests."""
    return {
        'material type': 'test material',
        'search fields': {
            'Name': {
                'prompt': 'Material name',
                'aliases': ['Compound name'],
            },
            'Conductivity': {
                'prompt': 'Ionic conductivity',
                'unit': 'S cm^-1',
                'aliases': ['sigma'],
            },
        },
    }


def test_validate_recipe_accepts_valid_recipe_and_adds_prompt_default():
    """
    Test validation of a valid recipe dictionary.

    This function performs the following steps:
    1. Builds a valid recipe without an `additional prompts` field.
    2. Validates the recipe with `_validate_recipe`.
    3. Checks the normalized recipe.

    Asserts:
        - The validated recipe keeps its original fields.
        - The missing `additional prompts` field is added as an empty string.
    """
    recipe = sample_recipe()
    recipe.pop('additional prompts', None)

    validated = recipes._validate_recipe(recipe, 'test')

    assert validated['material type'] == 'test material'
    assert validated['additional prompts'] == ''


def test_validate_recipe_rejects_invalid_recipe_shapes():
    """
    Test recipe validation errors for invalid recipe shapes.

    This function performs the following steps:
    1. Validates a non-dictionary recipe.
    2. Validates a dictionary missing required keys.
    3. Validates a recipe with no search fields.

    Asserts:
        - Non-dictionary recipes raise `ValueError`.
        - Recipes missing required keys raise `ValueError`.
        - Recipes without search fields raise `ValueError`.
    """
    with pytest.raises(ValueError, match='must be a JSON object'):
        recipes._validate_recipe([], 'test')

    with pytest.raises(ValueError, match='missing required key'):
        recipes._validate_recipe({'material type': 'test'}, 'test')

    with pytest.raises(ValueError, match='one or more search fields'):
        recipes._validate_recipe({'material type': 'test', 'search fields': {}}, 'test')


def test_load_recipe_file_accepts_direct_and_named_recipe_files(tmp_path):
    """
    Test loading standalone recipe JSON files.

    This function performs the following steps:
    1. Writes a direct recipe object to a temporary JSON file.
    2. Writes a named single-recipe object to another temporary JSON file.
    3. Loads both recipe files.

    Asserts:
        - Direct recipe objects are loaded.
        - Single named recipe objects are loaded.
    """
    direct_path = tmp_path / 'direct.json'
    named_path = tmp_path / 'named.json'
    direct_path.write_text(json.dumps(sample_recipe()))
    named_path.write_text(json.dumps({'custom': sample_recipe()}))

    assert recipes._load_recipe_file(direct_path)['material type'] == 'test material'
    assert recipes._load_recipe_file(named_path)['material type'] == 'test material'


def test_load_recipe_file_rejects_invalid_json_and_ambiguous_files(tmp_path):
    """
    Test recipe file loading errors.

    This function performs the following steps:
    1. Writes invalid JSON to a temporary file.
    2. Writes an ambiguous multi-recipe object to another temporary file.
    3. Attempts to load both files.

    Asserts:
        - Invalid JSON raises `ValueError`.
        - Ambiguous recipe files raise `ValueError`.
    """
    invalid_path = tmp_path / 'invalid.json'
    ambiguous_path = tmp_path / 'ambiguous.json'
    invalid_path.write_text('{bad json')
    ambiguous_path.write_text(json.dumps({'one': sample_recipe(), 'two': sample_recipe()}))

    with pytest.raises(ValueError, match='not valid JSON'):
        recipes._load_recipe_file(invalid_path)

    with pytest.raises(ValueError, match='single recipe object'):
        recipes._load_recipe_file(ambiguous_path)


def test_load_recipe_reads_files_bundled_recipes_and_reports_missing(monkeypatch, tmp_path):
    """
    Test public recipe loading from paths and bundled recipes.

    This function performs the following steps:
    1. Writes a temporary recipe file and loads it through `load_recipe`.
    2. Replaces the bundled recipes path with a temporary bundled recipe file.
    3. Loads a bundled recipe and requests a missing bundled recipe.

    Asserts:
        - Path-based recipes are loaded directly.
        - Bundled recipes are loaded by lower-case name.
        - Missing bundled recipes raise `KeyError`.
    """
    recipe_path = tmp_path / 'recipe.json'
    bundled_path = tmp_path / 'recipes.json'
    recipe_path.write_text(json.dumps(sample_recipe()))
    bundled_path.write_text(json.dumps({'demo': sample_recipe()}))
    monkeypatch.setattr(recipes, 'RECIPES_PATH', bundled_path)

    assert recipes.load_recipe(str(recipe_path))['material type'] == 'test material'
    assert recipes.load_recipe('DEMO')['material type'] == 'test material'

    with pytest.raises(KeyError, match='does not exist'):
        recipes.load_recipe('missing')


def test_load_recipe_reports_missing_or_invalid_bundled_recipe_file(monkeypatch, tmp_path):
    """
    Test errors from the bundled recipe file.

    This function performs the following steps:
    1. Points `RECIPES_PATH` at a missing file.
    2. Points `RECIPES_PATH` at an invalid JSON file.
    3. Calls `load_recipe` in both cases.

    Asserts:
        - Missing bundled recipe files raise `FileNotFoundError`.
        - Invalid bundled recipe JSON raises `ValueError`.
    """
    missing_path = tmp_path / 'missing.json'
    invalid_path = tmp_path / 'recipes.json'

    monkeypatch.setattr(recipes, 'RECIPES_PATH', missing_path)
    with pytest.raises(FileNotFoundError):
        recipes.load_recipe('demo')

    invalid_path.write_text('{bad json')
    monkeypatch.setattr(recipes, 'RECIPES_PATH', invalid_path)
    with pytest.raises(ValueError, match='not valid JSON'):
        recipes.load_recipe('demo')


def test_field_columns_builds_recipe_columns_and_respects_existing_columns():
    """
    Test output column construction for recipe fields.

    This function performs the following steps:
    1. Builds columns for a sample recipe.
    2. Builds columns when existing columns are supplied.
    3. Compares the results to expected column lists.

    Asserts:
        - Unit-bearing fields include units in square brackets.
        - Metadata fields are appended to new recipe columns.
        - Existing columns are returned unchanged.
    """
    columns = recipes.field_columns(sample_recipe())

    assert columns[:2] == ['Name', 'Conductivity [S cm^-1]']
    assert columns[-5:] == recipes.METADATA_FIELDS
    assert recipes.field_columns(sample_recipe(), existing_columns=['Existing']) == ['Existing']


def test_aliases_for_includes_fields_prompts_aliases_and_metadata_fields():
    """
    Test alias construction for recipe and metadata fields.

    This function performs the following steps:
    1. Builds aliases for a sample recipe.
    2. Reads aliases for recipe fields.
    3. Reads aliases for metadata fields.

    Asserts:
        - Field names are aliases.
        - Prompts are aliases.
        - Explicit aliases are aliases.
        - Metadata fields are aliases for themselves.
    """
    aliases = recipes.aliases_for(sample_recipe())

    assert aliases['Name'] == {'name', 'material name', 'compound name'}
    assert aliases['Conductivity'] == {'conductivity', 'ionic conductivity', 'sigma'}
    assert aliases['doi'] == {'doi'}


def test_canonical_match_maps_aliases_units_and_rejects_unknown_columns(monkeypatch):
    """
    Test canonical matching of incoming scrape columns.

    This function performs the following steps:
    1. Defines output columns for a sample recipe.
    2. Matches an explicit alias to a unit-bearing output column.
    3. Matches a direct metadata field and a fallback base-name field.
    4. Matches an unknown field.

    Asserts:
        - Aliases map to their canonical output column.
        - Metadata fields map directly.
        - Columns missing from aliases can still match by base name.
        - Unknown incoming columns return None.
    """
    recipe = sample_recipe()
    columns = recipes.field_columns(recipe)

    assert recipes.canonical_match(' sigma ', columns, recipe) == 'Conductivity [S cm^-1]'
    assert recipes.canonical_match('doi', columns, recipe) == 'doi'
    monkeypatch.setattr(recipes, 'aliases_for', lambda _: {'Extra Column': {'different alias'}})
    assert recipes.canonical_match('extra column', ['Extra Column [kg]'], recipe) == 'Extra Column [kg]'
    assert recipes.canonical_match('Unknown field', columns, recipe) is None
