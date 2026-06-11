from paperscraper.gpt import gpt_scrape, token_length
import json
import pandas as pd
import math
import os
from tqdm import tqdm
from PyPDF2 import PdfReader
from pathlib import Path

## Get module directory
MODULE_DIR = Path(__file__).resolve().parent

def load_recipe(recipe_name: str):
    """
    Load configuration information from the recipes JSON file.

    Args:
        recipe_name (str): The name of the recipe to load.

    Returns:
        dict: A dictionary containing the configuration information.
    """
    try:
        with open(str(MODULE_DIR / 'resources' / 'recipes.json'),'r') as f:
            recipes = json.load(f)
            recipe = recipes[recipe_name.lower()]
    except FileNotFoundError:
        raise FileNotFoundError('The recipes.json file is missing. Please reinstall PaperScraper.')
    except KeyError:
        raise KeyError(f'Recipe called "{recipe_name}" does not exist.')
    except:
        raise ValueError('The recipes.json file may be corrupted and cannot not be read. Please reinstall PaperScraper.')
    return recipe

# Get text from PDF
def pdf_reader(pdf_path: str):
    """
    Extract text from a PDF.

    Args:
        pdf_path (str): The path of the PDF to read.

    Returns:
        str: A string containing the PDF's text.
    """
    reader = PdfReader(pdf_path)
    text = ''
    for page in reader.pages:
        text += page.extract_text(0)
    return text

# Get materials from downloaded papers
def scrape_papers(papers_dir: str, papers_path: str='papers.csv', recipe: str='sse'):
    """
    Scrape materials from downloaded papers.

    Args:
        papers_dir (str): The path of the PDF to read.
        in_file (str): Filepath of the 'to scrape' papers CSV.
        out_file (str): Filepath of the scraped papers CSV.
        recipe (str): Recipe to use for defining search parameters.
    """
    files = os.listdir(papers_dir)
    papers_df = pd.read_csv(papers_path, index_col=0)
    if 'scraped' in papers_df['status'].value_counts().keys():
        print('Some scraped papers have not been stored. Would you like to re-scrape these?')
        decision = input('Yes (Y)/ No (N): ')
        if decision.lower() in ['y', 'yes']:
            papers_df['status'].replace('scraped','retrieved',inplace=True)
            os.remove('temp_scraped_materials.csv')
        else:
            print('Scrape cancelled: Please run ps_store before scraping.')
            exit()
    recipe = load_recipe(recipe)
    first_material = True
    retrieved_count = papers_df['status'].value_counts()['retrieved']
    with tqdm(total=retrieved_count, desc='Scraping Papers', colour='green') as pbar:
        for i, row in papers_df.iterrows():
            if row['status'] == 'stored':
                continue
            scopus_id = row['dc:identifier'].split(':')[-1]
            filenames = [file for file in files if scopus_id in file]
            if filenames == []:
                retrieved_count -= 1
                pbar.total = retrieved_count
                pbar.refresh()
                continue
            filename = papers_dir + '/' + filenames[0]
            if filename.split('.')[-1] == 'txt':
                with open(filename, 'r') as f:
                    text = f.read()
            elif filename.split('.')[-1] == 'pdf':
                text = pdf_reader(filename)
                index = text.lower().rfind('references')
                text = text[:index]
            coeff = token_length(text, 'gpt-5')/120000
            if coeff <= 1:
                texts = [text]
            else:
                coeff = math.ceil(coeff)
                char_per_split = math.ceil(len(text)/coeff)
                texts = []
                for split in range(coeff):
                    if (split+1)*char_per_split >= len(text):
                        texts.append(text[split*char_per_split:])
                    else:
                        texts.append(text[split*char_per_split:(split+1)*char_per_split])
            for text in texts:
                response = gpt_scrape(text, recipe)
                if response == 'None':
                    continue
                else:
                    materials = []
                    for material in response:
                        material['Scopus id'] = row['dc:identifier']
                        material['doi'] = row['prism:doi']
                        material['Publication date'] = row['prism:coverDate']
                        materials.append(material)
                    if materials in [[],'','""',"''"]:
                        continue
                    if first_material:
                        materials_df = pd.DataFrame(materials)
                        first_material = False
                    else:
                        for material in materials:
                            materials_df.loc[len(materials_df)] = material
                            materials_df.reset_index(drop=True, inplace=True)
                    materials_df.to_csv('temp_scraped_materials.csv')
            papers_df.loc[i,'status'] = 'scraped'
            papers_df.to_csv(papers_path)
            pbar.update(1)