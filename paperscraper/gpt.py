from paperscraper import SETTINGS
import json
import openai
import tiktoken
import re
import math

## Get API key
openai.api_key = SETTINGS.get('gpt_api_key')

## Get token length of prompt
def token_length(prompt, model):
    if type(prompt) != str:
        return []
    enc = tiktoken.encoding_for_model(model)
    num_tokens = len(enc.encode(prompt))
    return num_tokens

## Search text using GPT 
def gpt_query(messages, model='gpt-4-turbo-2024-04-09'):
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=4000,
    )
    return response.choices[0].message.content

def gpt_scrape(text, recipe):
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
    response = gpt_query(messages, 'gpt-4-turbo-2024-04-09').replace('\n', '')
    materials = re.findall(r'\{.*?\}', response)
    data = []
    for material_json in materials:
        material = json.loads(material_json)
        data.append(material)
    return data

def gpt_unit_conversion(values, field, unit):
    prompt = f'Convert the following values of {field} to {unit}. Each value should be returned as a decimal on a separate line. Do not include the units. If you are unsure how to do the conversion, just return the original value. If the given value is "None", return None'
    values_str = ''
    for value in values:
        values_str += f'{value}\n'
    coeff = token_length(values_str, 'gpt-4')/120000
    if coeff <= 1:
        values_strs = [values_str]
    else:
        coeff = math.ceil(coeff)
        char_per_split = math.ceil(len(values_str)/coeff)
        values_strs = []
        for split in range(coeff):
            index = values_str.find('\n',split*char_per_split)+2
            index_2 = values_str.find('\n',(split+1)*char_per_split)+2
            if index_2 >= len(values_str):
                values_strs.append(values_str[index:])
            else:
                values_strs.append(values_str[index:index_2])
    output = []
    for values_str in values_strs:
        messages=[
            {
                'role': 'system',
                'content': prompt
            },
            {'role': 'user', 'content': values_str},
        ]
        converted_values = gpt_query(messages,'gpt-4-turbo-2024-04-09').splitlines()
        for value in converted_values:
            output.append(value)
    return output