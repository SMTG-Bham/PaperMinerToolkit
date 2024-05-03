from paperscraper import SETTINGS
import json
import openai
import pandas as pd
import tiktoken
import math
import re
import os
from tqdm import tqdm
from PyPDF2 import PdfReader
from pathlib import Path
from monty.serialization import loadfn

## Get API key
openai.api_key = SETTINGS.get('gpt_api_key')

## Get module directory
MODULE_DIR = Path(__file__).resolve().parent

def load_recipe(recipe_name: str):
    """
    Load configuration information from a JSON file.

    Args:
        fname (str): The name of the JSON file to load.

    Returns:
        dict: A dictionary containing the configuration information.
    """
    with open(str(MODULE_DIR / 'resources' / 'recipes.json'),'r') as f:
        recipes = json.load(f)
    recipe = recipes[recipe_name.lower()]
    return recipe

## Split querys into smaller chunks
def token_length(text):
    if type(text) != str:
        return []
    enc = tiktoken.encoding_for_model("gpt-4")
    num_tokens = len(enc.encode(text))
    coeff = num_tokens/120000
    if coeff <= 1:
        return [text]
    else:
        coeff = math.ceil(coeff)
        char_per_split = math.ceil(len(text)/coeff)
        texts = []
        for split in range(coeff):
            if (split+1)*char_per_split >= len(text):
                texts.append(text[split*char_per_split:])
            else:
                texts.append(text[split*char_per_split:(split+1)*char_per_split])
        return texts

## Search text using GPT 
def gpt_query(messages):
    response = openai.chat.completions.create(
        model='gpt-4-turbo-2024-04-09',
        messages=messages,
        temperature=0,
        max_tokens=4000,
    )
    return response.choices[0].message.content


## Format GPT response
def response_formatter(response):
    response = response.replace('\n', '')
    electrolytes = re.findall(r'\{.*?\}', response)
    data = []
    for electrolyte in electrolytes:
        sse = json.loads(electrolyte)
        data.append(sse)
    return data


def pdf_reader(pdf):
        reader = PdfReader(pdf)
        # number_of_pages = len(reader.pages)
        text = ""
        for page in reader.pages:
            text += page.extract_text(0)
        return text

def gpt_scrape(text, recipe):
    recipe = load_recipe(recipe)
    material_type = recipe['material type']
    search_fields = recipe['search fields']
    prompt = f'Extract the following structured data for {material_type} materials from the user\'s paper:\n\n'
    for field in search_fields:
        desc = search_fields[field]['prompt']
        prompt += f'- {desc}\n'
    prompt += f'\nThe output should be in JSON format, with each field as a key and the extracted data as a value. Create a new JSON string for each additional {material_type} material found. If a field is not mentioned in the paper, set the value to "None". For example:\n\n'
    prompt += '{\n'
    for key in search_fields.keys():
        example = search_fields[key]['example']
        if type(example) == str:
            prompt += f'\t"{key}": "{example}"\n'
        else:
            prompt += f'\t"{key}": '+str(example)+'\n'
    prompt += '}\n\nThe keys you use should match those above exactly.'
    messages=[
        {
            'role': 'system',
            'content': prompt
        },
        {'role': 'user', 'content': text},
    ]
    response = gpt_query(messages)
    return response

def gpt_unit_conversion():
    pass

def scrape_papers(path, recipe='sse'):
    files = os.listdir(path)
    to_scrape = pd.read_csv('papers_to_scrape.csv', index_col=0)
    if os.path.isfile('papers_scraped.csv'):
        scraped_papers = pd.read_csv('papers_scraped.csv', index_col=0)
    else:
        scraped_papers = to_scrape.copy()
        scraped_papers.drop(scraped_papers.index, inplace=True)
    with tqdm(total=len(to_scrape), desc='Scraping Papers', colour='green') as pbar:
        for i, row in to_scrape.iterrows():
            scopus_id = row['dc:identifier'].split(':')[-1]
            filenames = [file for file in files if scopus_id in file]
            if filenames == []:
                pbar.update(1)
                continue
            filename = path + '/' + filenames[0]
            if filename.split('.')[-1] == 'txt':
                with open(filename, 'r') as f:
                    text = f.read()
            elif filename.split('.')[-1] == 'pdf':
                text = pdf_reader(filename)
                index = text.lower().rfind('references')
                text = text[:index]
            texts = token_length(text)
            for text in texts:
                response = gpt_scrape(text, recipe)
                if response == 'None':
                    print(response)
                else:
                    data = response_formatter(response)
                    materials = []
                    for material in data:
                        material['Scopus id'] = row['dc:identifier']
                        material['doi'] = row['prism:doi']
                        material['Publication date'] = row['prism:coverDate']
                        materials.append(material)
                    materials_df = pd.DataFrame(materials)
                    row_count=0
                    if os.path.isfile('materials.csv'):
                        with open('materials.csv','r') as output_file:
                            row_count = sum(1 for row in output_file)
                    materials_df.index += row_count-1
                    materials_df.to_csv('materials.csv', mode='a', header=not os.path.exists('materials.csv'))
            scraped_papers.loc[len(scraped_papers)] = row
            scraped_papers.drop(scraped_papers[scraped_papers['dc:identifier'] == row['dc:identifier']].index)
            pbar.update(1)
    scraped_papers.to_csv('papers_scraped.csv')