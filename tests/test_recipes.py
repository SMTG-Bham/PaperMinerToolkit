"""Unit tests for paperscraper.recipes.

This module tests recipe validation, loading bundled and file-based recipes,
building output columns, alias collection, and canonical column matching.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import paperscraper.recipes as recipes


def sample_recipe() -> dict[str, Any]:
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


def test_validate_recipe_accepts_valid_recipe_and_adds_prompt_default() -> None:
    """Test validation of a valid recipe dictionary."""
    recipe = sample_recipe()
    recipe.pop('additional prompts', None)

    validated = recipes._validate_recipe(recipe, 'test')

    assert validated['material type'] == 'test material'
    assert validated['additional prompts'] == ''


def test_validate_recipe_rejects_invalid_recipe_shapes() -> None:
    """Test recipe validation errors for invalid recipe shapes."""
    with pytest.raises(ValueError, match='must be a JSON object'):
        recipes._validate_recipe([], 'test')

    with pytest.raises(ValueError, match='missing required key'):
        recipes._validate_recipe({'material type': 'test'}, 'test')

    with pytest.raises(ValueError, match='one or more search fields'):
        recipes._validate_recipe({'material type': 'test', 'search fields': {}}, 'test')


def test_load_recipe_file_accepts_direct_and_named_recipe_files(tmp_path: Path) -> None:
    """Test loading standalone recipe JSON files."""
    direct_path = tmp_path / 'direct.json'
    named_path = tmp_path / 'named.json'
    direct_path.write_text(json.dumps(sample_recipe()))
    named_path.write_text(json.dumps({'custom': sample_recipe()}))

    assert recipes._load_recipe_file(direct_path)['material type'] == 'test material'
    assert recipes._load_recipe_file(named_path)['material type'] == 'test material'


def test_load_recipe_file_rejects_invalid_json_and_ambiguous_files(tmp_path: Path) -> None:
    """Test recipe file loading errors."""
    invalid_path = tmp_path / 'invalid.json'
    ambiguous_path = tmp_path / 'ambiguous.json'
    invalid_path.write_text('{bad json')
    ambiguous_path.write_text(json.dumps({'one': sample_recipe(), 'two': sample_recipe()}))

    with pytest.raises(ValueError, match='not valid JSON'):
        recipes._load_recipe_file(invalid_path)

    with pytest.raises(ValueError, match='single recipe object'):
        recipes._load_recipe_file(ambiguous_path)


def test_load_recipe_reads_files_bundled_recipes_and_reports_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test public recipe loading from paths and bundled recipes."""
    recipe_path = tmp_path / 'recipe.json'
    bundled_path = tmp_path / 'recipes.json'
    recipe_path.write_text(json.dumps(sample_recipe()))
    bundled_path.write_text(json.dumps({'demo': sample_recipe()}))
    monkeypatch.setattr(recipes, 'RECIPES_PATH', bundled_path)

    assert recipes.load_recipe(str(recipe_path))['material type'] == 'test material'
    assert recipes.load_recipe('DEMO')['material type'] == 'test material'

    with pytest.raises(KeyError, match='does not exist'):
        recipes.load_recipe('missing')


def test_load_recipe_reports_missing_or_invalid_bundled_recipe_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test errors from the bundled recipe file."""
    missing_path = tmp_path / 'missing.json'
    invalid_path = tmp_path / 'recipes.json'

    monkeypatch.setattr(recipes, 'RECIPES_PATH', missing_path)
    with pytest.raises(FileNotFoundError):
        recipes.load_recipe('demo')

    invalid_path.write_text('{bad json')
    monkeypatch.setattr(recipes, 'RECIPES_PATH', invalid_path)
    with pytest.raises(ValueError, match='not valid JSON'):
        recipes.load_recipe('demo')


def test_bundled_band_gap_recipe_uses_structured_lists_and_material_granularity() -> None:
    """Keep band-gap values structured while returning one record per material."""
    recipe = recipes.load_recipe('band_gap_validation')

    assert list(recipe['search fields']) == [
        'Material system',
        'Band gap',
        'All band gaps',
        'Cited literature band gaps',
    ]
    assert 'one record per distinct material' in recipe['additional prompts']
    assert 'Do not split one material' in recipe['additional prompts']
    assert 'Return [] for the whole response' in recipe['additional prompts']
    assert 'prior work' in recipe['additional prompts']
    assert 'general missing value \'None\'' in recipe['additional prompts']

    gap_fields = ['Band gap', 'All band gaps', 'Cited literature band gaps']
    item_keys = {'value', 'method_or_source', 'gap_type', 'conditions'}
    for field in gap_fields:
        example = recipe['search fields'][field]['example']
        assert isinstance(example, list)
        assert example
        assert all(set(item) == item_keys for item in example)

    aliases = recipes.aliases_for(recipe)
    for field, field_aliases in aliases.items():
        other_aliases = set().union(*(names for owner, names in aliases.items() if owner != field))
        assert field_aliases.isdisjoint(other_aliases)


def test_field_columns_builds_recipe_columns_and_respects_existing_columns() -> None:
    """Test output column construction for recipe fields."""
    columns = recipes.field_columns(sample_recipe())

    assert columns[:2] == ['Name', 'Conductivity [S cm^-1]']
    assert columns[-5:] == recipes.METADATA_FIELDS
    assert recipes.field_columns(sample_recipe(), existing_columns=['Existing']) == ['Existing']


def test_aliases_for_includes_fields_prompts_aliases_and_metadata_fields() -> None:
    """Test alias construction for recipe and metadata fields."""
    aliases = recipes.aliases_for(sample_recipe())

    assert aliases['Name'] == {'name', 'material name', 'compound name'}
    assert aliases['Conductivity'] == {'conductivity', 'ionic conductivity', 'sigma'}
    assert aliases['doi'] == {'doi'}


def test_canonical_match_maps_aliases_units_and_rejects_unknown_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test canonical matching of incoming scrape columns."""
    recipe = sample_recipe()
    columns = recipes.field_columns(recipe)

    assert recipes.canonical_match(' sigma ', columns, recipe) == 'Conductivity [S cm^-1]'
    assert recipes.canonical_match('doi', columns, recipe) == 'doi'
    monkeypatch.setattr(recipes, 'aliases_for', lambda _: {'Extra Column': {'different alias'}})
    assert recipes.canonical_match('extra column', ['Extra Column [kg]'], recipe) == 'Extra Column [kg]'
    assert recipes.canonical_match('Unknown field', columns, recipe) is None
