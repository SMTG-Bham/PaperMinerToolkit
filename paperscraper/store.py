from paperscraper.scrape import load_recipe
from paperscraper.gpt import gpt_unit_conversion
import pandas as pd
import os

def store_results(in_filepath='temp_scraped_materials.csv', out_filepath='materials.csv', unit_conversion=True, recipe='sse'):
    in_file = pd.read_csv(in_filepath)
    recipe = load_recipe(recipe)
    fields = recipe['search_fields'].keys()
    if os.path.isfile(out_filepath):
        out_file = pd.read_csv(out_filepath)
        columns=out_file.keys()
    else:
        columns = []
        for field in fields:
            if 'unit' in recipe['search_fields'][field]:
                unit = recipe['search_fields'][field]['unit']
                columns.append(f'{field} [{unit}]')
            else:
                columns.append(field)
        columns=columns

    out_data = pd.DataFrame()
    for series_name, series in in_file.items():
        matches = [field for field in columns if series_name in field]
        if len(matches) >= 2:
            print(f'{series_name} matched with multiple fields:')
            for i, match in enumerate(matches):
                i += 1
                print(f'{i})\t{match}')
            decision = input('Select the correct field by typing the number, or type N if none of them match: ')
            if decision.lower() in ["n", "none"]:
                matches = []
            index = int(decision)-1
            matches = [matches[index]]
        if len(matches) == 1:
            split_field = matches[0].split(' [')
            if len(split_field) == 2 and unit_conversion:
                print(f'{series_name} column was matched with {matches[0]} and will be converted.')
                split_field[1] -= ']'
                converted_series = gpt_unit_conversion(series, field=split_field[0], unit=split_field[1])
            else:
                print(f'{series_name} column was matched with {matches[0]} and will not be converted.')
                converted_series = series
            out_data[matches[0]] = converted_series
            columns.remove(matches[0])
        elif len(matches) == 0:
            raise RuntimeError(f'{series_name} did not match with a field in the recipe or output file.')
    temp_filename = 'temp_converted_materials.csv'
    out_data.to_csv(temp_filename)
    # ask if user is happy with the conversions
    print(f'Output data has been saved to {temp_filename} temporarily. Are you happy with these conversions?')
    decision = input("Yes (Y)/ No (N): ")
    if decision.lower() in ["y", "yes"]:
        out_data = pd.read_csv(temp_filename)
        out_data.to_csv('materials.csv', mode='a', header=not os.path.exists('materials.csv'))
        os.remove(in_filepath)
    os.remove(temp_filename)
    
    
    