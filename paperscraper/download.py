from paperscraper import SETTINGS
from elsapy.elsclient import ElsClient
from elsapy.elsdoc import FullDoc
import json
import os
import pandas as pd
from tqdm import tqdm

## Get Elsevier API key and initialize client
client = ElsClient(SETTINGS.get('elsevier_api_key'))

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
    if type(text) == dict:
        return 'failed'
    return text

## Remove unnecessary text from ScienceDirect (full-text) strings
def elsevier_string_formatter(text: str):
    if text.count('Acknowledgements') == 2:
        text = text.split('Acknowledgements')[1]
    elif text.count('References') == 2:
        text = text.split('References')[1]
    if 'amazonaws.com/' in text:
        text = text.split('amazonaws.com/')[-1]
        text = text[text.find(' '):]
    return text


## Download ScienceDirect (full-text) documents using URIs
def elsevier_downloader(papers_path='papers.csv', download_dir='papers'):
    if not os.path.isdir(download_dir):
        os.mkdir(download_dir)
    papers = pd.read_csv(papers_path)
    elsevier_papers = papers[papers['link'].str.contains('full-text')]
    with tqdm(total=len(elsevier_papers['link']), desc='Downloading Papers', colour='#A020F0') as pbar:

        a=0

        for i, paper in elsevier_papers.iterrows():
            filename = paper['dc:identifier'].split(':')[-1]
            filepath = f'{download_dir}/{filename}.txt'
            if os.path.isfile(filepath):
                pass
            else:
                uri = paper['link'].split("'")[-2]
                retrieve_document(uri)
                file = os.listdir('data')
                if len(file) < 1:
                    continue
                file = 'data/' + file[0]
                text = json_to_text(file)
                formatted_text = elsevier_string_formatter(text)
                with open(filepath,'w') as out_file:
                    out_file.write(formatted_text)
            pbar.update(1)

            a+=1
            if a == 100:
                break # remove when done testing