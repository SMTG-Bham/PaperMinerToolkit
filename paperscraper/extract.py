import json
import math
import re
from json import JSONDecodeError

import tiktoken

from paperscraper.models import ModelConfig, query_images, query_text
from paperscraper.settings import DEFAULT_MODEL



def token_length(prompt, model=DEFAULT_MODEL):
    if type(prompt) != str:
        return []
    try:
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(prompt))
    except Exception:
        return max(1, math.ceil(len(prompt) / 4))


def build_scrape_prompt(recipe, source='paper'):
    material_type = recipe['material type']
    search_fields = recipe['search fields']
    additional_prompts = recipe['additional prompts']
    prompt = f"Extract the following structured data for {material_type} materials from the user's {source}:\n\n"
    for field in search_fields:
        desc = search_fields[field]['prompt']
        prompt += f'- {desc}\n'
    prompt += f'\nThe output should be in JSON format, with each field as a key and the extracted data as a value. Create a new JSON string for each additional {material_type} material found. Be critical of your decisions and review your answers to make sure they reflect the information in the {source}. If there are multiple materials or permutations of materials, make sure to scrape all of them. If a field is not mentioned, set the value to "None". Only put values in square brackets if multiple values are matched with the field. {additional_prompts} For example:\n\n'
    prompt += '{\n'
    for key in search_fields.keys():
        example = search_fields[key]['example']
        if type(example) == str:
            prompt += f'\t"{key}": "{example}"\n'
        else:
            prompt += f'\t"{key}": ' + str(example) + '\n'
    prompt += '}\n\nThe keys you use should match those above exactly. Return only JSON objects and no explanatory text.'
    return prompt


def query_model(messages, model_config=None):
    config = model_config or ModelConfig.from_profile('text')
    return query_text(messages, config=config, max_output_tokens=10000)


def _strip_fences(response):
    response = response.strip()
    fence = chr(96) * 3
    if response.startswith(fence):
        response = re.sub(r'^' + fence + r'(?:json)?', '', response, flags=re.IGNORECASE).strip()
        response = re.sub(fence + r'$', '', response).strip()
    return response


def _json_decoder_scan(response):
    decoder = json.JSONDecoder()
    data = []
    index = 0
    while index < len(response):
        positions = [pos for pos in (response.find('{', index), response.find('[', index)) if pos != -1]
        if not positions:
            break
        start = min(positions)
        try:
            parsed, end = decoder.raw_decode(response[start:])
        except JSONDecodeError:
            index = start + 1
            continue
        if isinstance(parsed, list):
            data.extend(item for item in parsed if isinstance(item, dict))
        elif isinstance(parsed, dict):
            data.append(parsed)
        index = start + end
    return data


def _extract_json_objects(response):
    response = _strip_fences(response)
    data = _json_decoder_scan(response)
    if data:
        return data
    compact = response.replace('\n', '')
    for material_json in re.findall(r'\{[^{}]*\}', compact, flags=re.DOTALL):
        try:
            data.append(json.loads(material_json))
        except JSONDecodeError:
            continue
    if data:
        return data
    preview = response[:500].replace('\n', ' ')
    raise ValueError(f'Model response did not contain valid JSON objects. Response preview: {preview}')


def scrape_text(text, recipe, model_config=None):
    prompt = build_scrape_prompt(recipe, source='paper')
    messages = [
        {'role': 'system', 'content': prompt},
        {'role': 'user', 'content': text},
    ]
    response = query_model(messages, model_config=model_config)
    return _extract_json_objects(response)


def scrape_images(image_paths, recipe, model_config=None, context=None):
    config = model_config or ModelConfig.from_profile('vision')
    prompt = build_scrape_prompt(recipe, source='paper image')
    response = query_images(prompt, image_paths, config=config, context=context, max_output_tokens=10000)
    return _extract_json_objects(response)


def scrape_pdf(filepath, recipe, model_config=None):
    from paperscraper.documents import read_pdf_text

    return scrape_text(read_pdf_text(filepath), recipe, model_config=model_config)


def convert_units(values, field, unit, model_config=None):
    prompt = f'Convert the following values of {field} to {unit}. Each result should be returned as a decimal on a separate line. If the input contains multiple values on one line, return the converted values as a python list on the same line. Only put values in square brackets if multiple values are provided on the line. Do not include the units. If you are unsure how to do the conversion, just return the original value. If a range is given, report this as two decimals with a hyphen/dash inbetween (For example: 1-10). If the value is already in the desired unit, just convert it to a decimal. Do not return "None". Do not return the value as an addition. If text is given and cannot be meaningfully converted, return the same text. Convert "RT" or "Room temperature" to the equivalent of 298.15K. Do not use quotation marks. Make sure that there are as many output values as input.'
    values_str = ''
    memory = []
    for value in values:
        if str(value) in ['nan', "['None']"]:
            memory.append(0)
        else:
            memory.append(1)
            values_str += f'{value}\n'
    config = model_config or ModelConfig.from_profile('text')
    coeff = token_length(values_str, config.name) / 200000
    if coeff <= 1:
        values_strs = [values_str]
    else:
        coeff = math.ceil(coeff)
        char_per_split = math.ceil(len(values_str) / coeff)
        values_strs = []
        for split in range(coeff):
            index = values_str.find('\n', split * char_per_split) + 2
            index_2 = values_str.find('\n', (split + 1) * char_per_split) + 2
            if index_2 >= len(values_str):
                values_strs.append(values_str[index:])
            else:
                values_strs.append(values_str[index:index_2])
    output = []
    temp = []
    if values_strs[0] != '':
        for values_str in values_strs:
            messages = [
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': values_str},
            ]
            converted_values = query_model(messages, model_config=config).splitlines()
            for value in converted_values:
                temp.append(value)
    index = 0
    for mem in memory:
        if mem == 0:
            output.append(None)
        elif mem == 1:
            output.append(temp[index])
            index += 1
    return output
