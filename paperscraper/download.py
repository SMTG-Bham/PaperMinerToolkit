from elsapy.elsclient import ElsClient
from elsapy.elsdoc import FullDoc
import json
import os
import pandas as pd
from tqdm import tqdm

## Load configuration
with open("/rds/homes/o/ogs353/sse-project/JLR/experimental_database/config.json", 'r') as con_file:
    config = json.load(con_file)

## Initialize client
client = ElsClient(config['elsevier_api_key'])


## ScienceDirect (full-text) documents using URIs
def retrieve_document(uri):
    files = os.listdir('data')
    for file in files:
        os.remove('data/' + file)
    doi_doc = FullDoc(uri=uri)
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

## Remove unnecessary text from ScienceDirect (full-text) strings
def elsevier_string_formatter(text):
    if text.count('Acknowledgements') == 2:
        text = text.split('Acknowledgements')[1]
    elif text.count('References') == 2:
        text = text.split('References')[1]
    if 'amazonaws.com/' in text:
        text = text.split('amazonaws.com/')[-1]
        text = text[text.find(' '):]
    return text


## Download ScienceDirect (full-text) documents using URIs
def elsevier_downloader(papers_path='papers_to_scrape.csv', download_path='papers'):
    papers = pd.read_csv(papers_path)
    elsevier_papers = papers[papers['link'].str.contains('full-text')]
    with tqdm(total=len(elsevier_papers['link']), desc='Downloading Papers', colour='#A020F0') as pbar:

        a=0

        for paper in elsevier_papers.itertuples():
            uri = paper[3].split("'")[-2]
            pbar.update(1)
            retrieve_document(uri)
            file = os.listdir('data')
            file = 'data/' + file[0]
            text = json_to_text(file)
            formatted_text = elsevier_string_formatter(text)

            filename = paper[5].split(':')[-1]
            with open(f'{download_path}/{filename}.txt','w') as out_file:
                out_file.write(formatted_text)

            a+=1
            if a == 10:
                break # remove when done testing