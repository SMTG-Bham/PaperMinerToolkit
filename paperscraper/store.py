from paperscraper.scrape import load_recipe
from paperscraper.gpt import gpt_unit_conversion
import pandas as pd
import os

def store_results(papers_path='papers.csv', in_filepath='temp_scraped_materials.csv', out_filepath='materials.csv', unit_conversion=True, recipe='sse'):
    in_file = pd.read_csv(in_filepath, index_col=0)
    recipe = load_recipe(recipe)
    fields = recipe['search fields'].keys()
    if os.path.isfile(out_filepath):
        out_file = pd.read_csv(out_filepath, index_col=0)
        columns=list(out_file.keys())
    else:
        columns = []
        for field in fields:
            if 'unit' in recipe['search fields'][field]:
                unit = recipe['search fields'][field]['unit']
                columns.append(f'{field} [{unit}]')
            else:
                columns.append(field)
        columns=list(columns)
    out_data = pd.DataFrame()
    for series_name, series in in_file.items():
        matches = [field for field in columns if series_name in field]
        if len(matches) == 0:
            matches = [string for string in ['Scopus id', 'doi', 'Publication date'] if string in series_name]
            if len(matches) == 0:
                raise RuntimeError(f'{series_name} did not match with a field in the recipe or output file.')
        if len(matches) >= 2:
            print(f'{series_name} matched with multiple fields:')
            for i, match in enumerate(matches):
                i += 1
                print(f'{i})\t{match}')
            decision = input('Select the correct field by typing the number, or type N if none of them match: ')
            if decision.lower() in ["n", "none"]:
                raise RuntimeError(f'{series_name} did not match with a field in the recipe or output file.')
            index = int(decision)-1
            matches = [matches[index]]
        if len(matches) == 1:
            split_field = matches[0].split(' [')
            if len(split_field) == 2 and unit_conversion:
                print(f'{series_name} column was matched with {matches[0]} and will be converted.')
                split_field[1] = split_field[1].replace(']','')
                converted_series = gpt_unit_conversion(series, field=split_field[0], unit=split_field[1])
            else:
                print(f'{series_name} column was matched with {matches[0]} and will not be converted.')
                converted_series = series
            out_data[matches[0]] = converted_series
            if not matches[0] in ['Scopus id', 'doi', 'Publication date']:
                try:
                    columns.remove(matches[0])
                except Exception as e:
                    print(matches[0], e) # add error message
    temp_filename = 'temp_converted_materials.csv'
    out_data.to_csv(temp_filename)
    print(f'\nOutput data has been saved to {temp_filename} temporarily. Are you happy with these conversions?')
    decision = input("Yes (Y) / No (N): ")
    if decision.lower() in ["y", "yes"]:
        out_data = pd.read_csv(temp_filename, index_col=0)
        if os.path.isfile(out_filepath):
            materials_df = pd.read_csv(out_filepath, index_col=0)
            materials_df = pd.concat([materials_df, out_data], ignore_index=True)
            materials_df.reset_index(drop=True, inplace=True)
        else:
            materials_df = out_data
        materials_df.to_csv(out_filepath)
        os.remove(in_filepath)
        papers_df = pd.read_csv(papers_path, index_col=0)
        papers_df['status'].replace('scraped','stored',inplace=True)
        papers_df.to_csv(papers_path)
    os.remove(temp_filename)