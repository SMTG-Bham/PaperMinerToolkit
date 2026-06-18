import json
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
RECIPES_PATH = MODULE_DIR / 'resources' / 'recipes.json'
METADATA_FIELDS = ['Scopus id', 'doi', 'Publication date', 'Source', 'Source path']


def load_recipe(recipe_name: str):
    try:
        with open(RECIPES_PATH, 'r', encoding='utf-8') as f:
            recipes = json.load(f)
            return recipes[recipe_name.lower()]
    except FileNotFoundError:
        raise FileNotFoundError('The recipes.json file is missing. Please reinstall PaperScraper.')
    except KeyError as e:
        raise KeyError(f'Recipe called "{recipe_name}" does not exist.') from e
    except Exception as e:
        raise ValueError('The recipes.json file may be corrupted and cannot not be read. Please reinstall PaperScraper.') from e


def field_columns(recipe, existing_columns=None):
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
