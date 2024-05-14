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
def scrape_papers(papers_path: str, in_file: str='papers_to_scrape.csv', out_file: str='papers_scraped.csv', recipe: str='sse'):
    """
    Scrape materials from downloaded papers.

    Args:
        papers_path (str): The path of the PDF to read.
        in_file (str): Filepath of the 'to scrape' papers CSV.
        out_file (str): Filepath of the scraped papers CSV.
        recipe (str): Recipe to use for defining search parameters.
    """
    files = os.listdir(papers_path)
    to_scrape = pd.read_csv(in_file, index_col=0)
    if os.path.isfile(out_file):
        scraped_papers = pd.read_csv(out_file, index_col=0)
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
            filename = papers_path + '/' + filenames[0]
            if filename.split('.')[-1] == 'txt':
                with open(filename, 'r') as f:
                    text = f.read()
            elif filename.split('.')[-1] == 'pdf':
                text = pdf_reader(filename)
                index = text.lower().rfind('references')
                text = text[:index]
            coeff = token_length(text, 'gpt-4')/120000
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
            recipe = load_recipe(recipe)
            for text in texts:
                response = gpt_scrape(text, recipe)
                if response == 'None':
                    print(response)
                else:
                    materials = []
                    for material in response:
                        material['Scopus id'] = row['dc:identifier']
                        material['doi'] = row['prism:doi']
                        material['Publication date'] = row['prism:coverDate']
                        materials.append(material)
                    materials_df = pd.DataFrame(materials)
                    row_count=0
                    if os.path.isfile('temp_scraped_materials.csv'):
                        with open('temp_scraped_materials.csv','r') as output_file:
                            row_count = sum(1 for row in output_file)
                    materials_df.index += row_count
                    materials_df.to_csv('temp_scraped_materials.csv', mode='a', header=not os.path.exists('temp_scraped_materials.csv'))
            scraped_papers.loc[len(scraped_papers)] = row
            scraped_papers.drop(scraped_papers[scraped_papers['dc:identifier'] == row['dc:identifier']].index, inplace=True)
            pbar.update(1)
    scraped_papers.to_csv(out_file)