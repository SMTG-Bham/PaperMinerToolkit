from paperscraper.scrape import load_recipe
from paperscraper.gpt import gpt_unit_conversion
from paperscraper.pipeline import ensure_pipeline_columns, write_papers
import pandas as pd
import os

METADATA_FIELDS = ['Scopus id', 'doi', 'Publication date', 'Source', 'Source path']


def _field_columns(recipe, existing_columns=None):
    columns = list(existing_columns or [])
    if not columns:
        for field, config in recipe['search fields'].items():
            if 'unit' in config:
                columns.append(f'{field} [{config["unit"]}]')
            else:
                columns.append(field)
        columns.extend(METADATA_FIELDS)
    return columns


def _aliases_for(recipe):
    aliases = {}
    for field, config in recipe['search fields'].items():
        names = {field.lower()}
        for alias in config.get('aliases', []):
            names.add(alias.lower())
        aliases[field] = names
    for field in METADATA_FIELDS:
        aliases[field] = {field.lower()}
    return aliases


def _canonical_match(series_name, columns, recipe):
    raw_name = series_name.strip()
    normalized = raw_name.lower()
    aliases = _aliases_for(recipe)
    for column in columns:
        base = column.split(' [')[0]
        if normalized in aliases.get(base, {base.lower()}):
            return column
    for column in columns:
        base = column.split(' [')[0]
        if normalized == base.lower():
            return column
    return None


def store_results(
    papers_path='papers.csv',
    in_filepath='temp_scraped_materials.csv',
    out_filepath='materials.csv',
    unit_conversion=True,
    recipe='sse',
    assume_yes=False,
    model_config=None,
):
    in_file = pd.read_csv(in_filepath, index_col=0)
    recipe = load_recipe(recipe)
    if os.path.isfile(out_filepath):
        out_file = pd.read_csv(out_filepath, index_col=0)
        columns = list(out_file.keys())
    else:
        columns = _field_columns(recipe)

    out_data = pd.DataFrame()
    for series_name, series in in_file.items():
        match = _canonical_match(series_name, columns, recipe)
        if match is None:
            if assume_yes:
                print(f'Skipping unmatched column in noninteractive mode: {series_name}')
                continue
            raise RuntimeError(f'{series_name} did not match with a field in the recipe or output file.')
        split_field = match.split(' [')
        if len(split_field) == 2 and unit_conversion:
            print(f'{series_name} column was matched with {match} and will be converted.')
            split_field[1] = split_field[1].replace(']', '')
            converted_series = gpt_unit_conversion(
                series,
                field=split_field[0],
                unit=split_field[1],
                model_config=model_config,
            )
        else:
            print(f'{series_name} column was matched with {match} and will not be converted.')
            converted_series = series
        out_data[match] = converted_series

    temp_filename = 'temp_converted_materials.csv'
    out_data.to_csv(temp_filename)
    if assume_yes:
        decision = 'yes'
    else:
        print(f'\nOutput data has been saved to {temp_filename} temporarily. Are you happy with these conversions?')
        decision = input('Yes (Y) / No (N): ')
    if decision.lower() in ['y', 'yes']:
        out_data = pd.read_csv(temp_filename, index_col=0)
        if os.path.isfile(out_filepath):
            materials_df = pd.read_csv(out_filepath, index_col=0)
            materials_df = pd.concat([materials_df, out_data], ignore_index=True)
            materials_df.reset_index(drop=True, inplace=True)
        else:
            materials_df = out_data
        materials_df.to_csv(out_filepath)
        os.remove(in_filepath)
        papers_df = ensure_pipeline_columns(pd.read_csv(papers_path, index_col=0))
        papers_df['store_status'] = papers_df['store_status'].where(
            ~((papers_df['text_scrape_status'] == 'succeeded') | (papers_df['image_scrape_status'] == 'succeeded')),
            'stored',
        )
        write_papers(papers_df, papers_path)
    os.remove(temp_filename)
