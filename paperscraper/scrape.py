import json
import openai
import pandas as pd
import tiktoken
import math
import re
import os
from tqdm import tqdm
from PyPDF2 import PdfReader
    
## Load configuration
with open("/rds/homes/o/ogs353/sse-project/JLR/experimental_database/config.json", 'r') as con_file:
    config = json.load(con_file)

## Get API key
openai.api_key = config['gpt_api_key']


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
def gpt_query(text):
    response = openai.chat.completions.create(
        model='gpt-4-turbo-2024-04-09',
        messages=[
            {
                "role": "system",
                "content": """
Extract the following structured data for solid electrolytes from the user's paper:

- Name of electrolyte
- Chemical formula
- Dopants
- Coatings
- Impurity phases
- Structure type
- Space group
- Electronic conductivity
- Ionic conductivity (if multiple measurements were taken, provide a list separated by commas)
- Activation energy for migration
- Hardness
- Young’s modulus
- Shear modulus
- Bulk modulus
- Crystal density
- Surface area
- Porosity
- Particle size
- Temperature (if multiple measurements were taken, list these in the same order as the conductivities)
- Pressure
- Pellet thickness
- Contact area
- Cell format
- Pellet density
- Critical current density for plating/stripping
- Coulombic efficiency
- Cycle life
- AC frequency range
- Synthesis conditions
- Type of study (experimental, computational, or both)

The output should be in JSON format, with each field as a key and the extracted data as a value. Create a new JSON string for each additional electrolyte found. If a field is not mentioned in the paper, set the value to "None". For example: 

{
    "Name": "LLZO",
    "Formula": "Li7La3Zr2O12",
    "Dopants": "Al",
    "Coatings": "None",
    "Impurity phases": "Amorphous Li2O",
    "Structure type": "Garnet",
    "Space group": " I-43d",
    "Electronic conductivity": "None",
    "Ionic conductivity": ["2.0 × 10−4 S cm−1","3.0 × 10−4 S cm−1"],
    "Activation energy": "0.25 eV",
    "Hardness": "9.1 GPa",
    "Young's modulus": "162.6 GPa",
    "Shear modulus": "64.6 GPa",
    "Bulk modulus": "112.4 GPa",
    "Crystal density": "None",
    "Surface area": "None",
    "Porosity": "None",
    "Particle size": "300nm",
    "Temperature": ["30 °C", "60 °C"],
    "Pressure": "Ambient",
    "Pellet thickness": "0.5 mm",
    "Contact area": "None",
    "Cell format": "None",
    "Pellet density": "None",
    "Critical current density": "None",
    "Coulombic efficiency" "None",
    "Cycle life": "None",
    "AC frequency range": "None",
    "Synthesis conditions": "Ball milled, Calcined at 850 °C, Sintered at 1180 °C",
    "Type of study": "Experimental"
}

The keys you use should match those above exactly.
"""
            },
            {"role": "user", "content": text},
        ],
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


def scrape_papers(path):
    files = os.listdir(path)
    to_scrape = pd.read_csv('papers_to_scrape.csv',index_col=0)
    if os.path.isfile('papers_scraped.csv'):
        scraped_papers = pd.read_csv('papers_scraped.csv',index_col=0)
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
                response = gpt_query(text)
                if response == 'None':
                    print(response)
                else:
                    data = response_formatter(response)
                    electrolytes = []
                    for electrolyte in data:
                        electrolyte['Scopus id'] = row['dc:identifier']
                        electrolyte['doi'] = row['prism:doi']
                        electrolyte['Publication date'] = row['prism:coverDate']
                        electrolytes.append(electrolyte)
                    electrolytes_df = pd.DataFrame(electrolytes)
                    row_count=0
                    if os.path.isfile('electrolytes.csv'):
                        with open('electrolytes.csv','r') as output_file:
                            row_count = sum(1 for row in output_file)
                    electrolytes_df.index += row_count-1
                    electrolytes_df.to_csv('electrolytes.csv', mode='a', header=not os.path.exists('electrolytes.csv'))
            scraped_papers.loc[len(scraped_papers)] = row
            # REMOVE PAPER FROM TO SCRAPE
            pbar.update(1)
    scraped_papers.to_csv('papers_scraped.csv')