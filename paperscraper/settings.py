from elsapy.elsclient import ElsClient
import openai
import requests
import json
import os

SETTINGS_FILE = os.path.join(os.path.expanduser('~'), '.config', '.pscraperrc.json')

def check_openai_api_key(api_key):
    client = openai.OpenAI(api_key=api_key)
    try:
        client.models.list()
    except openai.AuthenticationError:
        return False
    else:
        return True

def update_openai_key(settings=True):
    if settings:
        with open(SETTINGS_FILE, mode='r', encoding="utf-8") as json_file:
            settings = json.load(json_file)
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

def update_elsevier_key(settings=True):
    if settings:
        with open(SETTINGS_FILE, mode='r', encoding="utf-8") as json_file:
            settings = json.load(json_file)
    api_key = input('Enter Elsevier API key: ')
    if check_elsevier_api_key(api_key):
        settings['elsevier_api_key'] = api_key
        with open(SETTINGS_FILE, mode='w', encoding="utf-8") as json_file:
            json.dump(settings, json_file)
    else:
        raise ValueError('Elsevier API key is invalid.')