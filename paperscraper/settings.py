from paperscraper import SETTINGS_FILE, SETTINGS
from elsapy.elsclient import ElsClient
import openai
import requests
import json

def check_openai_api_key(api_key):
    client = openai.OpenAI(api_key=api_key)
    try:
        client.models.list()
    except openai.AuthenticationError:
        return False
    else:
        return True

def update_openai_key(settings=SETTINGS):
    api_key = input('Enter OpenAI API key: ')
    if check_openai_api_key(api_key):
        settings['openai_api_key'] = api_key
        with open(SETTINGS_FILE, mode='w', encoding="utf-8") as json_file:
            json.dump(settings, json_file)
    else:
        raise ValueError('OpenAI API key is invalid.')

def check_elsevier_api_key(api_key):
    client = ElsClient(api_key)
    try:
        url = u'https://api.elsevier.com/content/search/scopus?query=Test&count=1'
        api_response = client.exec_request(url)
    except requests.HTTPError:
        return False
    else:
        return True

def update_elsevier_key(settings=SETTINGS):
    api_key = input('Enter Elsevier API key: ')
    if check_elsevier_api_key(api_key):
        settings['elsevier_api_key'] = api_key
        with open(SETTINGS_FILE, mode='w', encoding="utf-8") as json_file:
            json.dump(settings, json_file)
    else:
        raise ValueError('Elsevier API key is invalid.')