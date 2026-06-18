from paperscraper.extract import convert_units
from paperscraper.pipeline import ensure_pipeline_columns, write_papers
from paperscraper.recipes import canonical_match, field_columns, load_recipe
import pandas as pd
import os


def store_results(
    papers_path='papers.csv',
    in_filepath='temp_scraped_materials.csv',
    out_filepath='materials.csv',
    unit_conversion=True,
    recipe='sse',
    assume_yes=False,
    model_config=None,
):
    if not os.path.isfile(in_filepath):
        print(f'No scraped materials file found at {in_filepath}. Nothing new to store.')
        return
    in_file = pd.read_csv(in_filepath, index_col=0)
    if in_file.empty:
        print(f'Scraped materials file {in_filepath} is empty. Nothing new to store.')
        return
    recipe = load_recipe(recipe)
    if os.path.isfile(out_filepath):
        out_file = pd.read_csv(out_filepath, index_col=0)
        columns = list(out_file.keys())
    else:
        columns = field_columns(recipe)

    out_data = pd.DataFrame()
    for series_name, series in in_file.items():
        match = canonical_match(series_name, columns, recipe)
        if match is None:
            if assume_yes:
                print(f'Skipping unmatched column in noninteractive mode: {series_name}')
                continue
            raise RuntimeError(f'{series_name} did not match with a field in the recipe or output file.')
        split_field = match.split(' [')
        if len(split_field) == 2 and unit_conversion:
            print(f'{series_name} column was matched with {match} and will be converted.')
            split_field[1] = split_field[1].replace(']', '')
            converted_series = convert_units(
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
