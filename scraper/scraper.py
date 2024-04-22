from elsapy.elsclient import ElsClient
from elsapy.elssearch import ElsSearch
from elsapy.elsdoc import FullDoc
import json
import openai
import os
import pandas as pd
import tiktoken
import math
import re
    
## Load configuration
with open("/rds/homes/o/ogs353/sse-project/JLR/experimental_database/config.json", 'r') as con_file:
    config = json.load(con_file)

## Initialize client
client = ElsClient(config['elsevier_api_key'])
openai.api_key = config['gpt_api_key']


## Initialize doc search object using ScienceDirect and execute search, retrieving all results
def document_search(query, index='sciencedirect'):
    doc_srch = ElsSearch(query, index)
    doc_srch.execute(client, get_all = True)
    print ("doc_srch has", len(doc_srch.results), "results.")

    return doc_srch.results_df


## ScienceDirect (full-text) documents using DOIs
def retrieve_document(doi):
    files = os.listdir('data')
    for file in files:
        os.remove('data/' + file)
    doi_doc = FullDoc(doi = doi)
    if doi_doc.read(client):
        doi_doc.write()
    else:
        print ("Read document failed.")


## Convert ScienceDirect (full-text) JSON to string
def json_to_text(filepath):
    with open(filepath, "r") as f:
        doc = json.load(f)
    text = doc["originalText"]
    return text

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
    print ('Sending Query')

    response = openai.chat.completions.create(
        model="gpt-4-0125-preview",
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
    print ('Response Recieved')
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


if __name__ == '__main__':
    papers = document_search('Lithium solid electrolyte')
    papers.to_csv('unscraped_papers.csv')
    for i, row in papers.iterrows():
        print (i)



        if i == 5:
            break

        if os.path.isfile('scraped_papers.txt'):
            with open('scraped_papers.txt') as f:
                if row['prism:doi'] in f.read():
                    print('true')
                    continue

        retrieve_document(row['prism:doi'])
        file = os.listdir('data')
        text = json_to_text('data/' + file[0])
        texts = token_length(text)
        for text in texts:
            response = gpt_query(text)
            if response == 'None':
                print(response)
            else:
                data = response_formatter(response)
                print (data)
                electrolytes = []
                for electrolyte in data:
                    electrolyte['doi'] = row['prism:doi']
                    electrolyte['Publication date'] = row['prism:coverDate']
                    electrolytes.append(electrolyte)
                electrolytes_df = pd.DataFrame(electrolytes)
                row_count=0
                if os.path.isfile('electrolytes_GPT4_new.csv'):
                    with open('electrolytes_GPT4_new.csv','r') as output_file:
                        row_count = sum(1 for row in output_file)
                electrolytes_df.index += row_count-1
                electrolytes_df.to_csv('electrolytes_GPT4_new.csv', mode='a', header=not os.path.exists('electrolytes_GPT4_new.csv'))
        with open('scraped_papers.txt','a') as scraped_papers:
            scraped_papers.write(row['prism:doi']+'\n')
    