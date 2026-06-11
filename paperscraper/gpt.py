from paperscraper import SETTINGS
import json
import openai
import tiktoken
import re
import math

## Get API key
openai.api_key = SETTINGS.get('openai_api_key')

## Set model (not implemented)
model = 'gpt-5-mini'

## Get token length of prompt
def token_length(prompt, model):
    if type(prompt) != str:
        return []
    enc = tiktoken.encoding_for_model(model)
    num_tokens = len(enc.encode(prompt))
    return num_tokens

## Search text using GPT 
def gpt_query(messages, model='gpt-5-mini'):
    try:
        response = openai.responses.create(
            model=model,
            input=messages,
            max_output_tokens=10000,
            service_tier='flex',
            reasoning={'effort': 'medium'}
        )
    except openai.BadRequestError as e:
        # Handle error 400
        raise Exception(f'OpenAI Error 400: {e}')
    except openai.AuthenticationError as e:
        # Handle error 401
        raise Exception(f'OpenAI Error 401: {e}')
    except openai.PermissionDeniedError as e:
        # Handle error 403
        raise Exception(f'OpenAI Error 403: {e}')
    except openai.NotFoundError as e:
        # Handle error 404
        raise Exception(f'OpenAI Error 404: {e}')
    except openai.UnprocessableEntityError as e:
        # Handle error 422
        raise Exception(f'OpenAI Error 422: {e}')
    except openai.RateLimitError as e:
        # Handle error 429
        raise Exception(f'OpenAI Error 429: {e}')
    except openai.InternalServerError as e:
        # Handle error >=500
        raise(f'OpenAI >=500: {e}')
    except openai.APIConnectionError as e:
        # Handle API connection error
        raise(f'OpenAI API connection error: {e}')
    return response.output[1].content[0].text

def gpt_scrape(text, recipe):
    material_type = recipe['material type']
    search_fields = recipe['search fields']
    additional_prompts = recipe['additional prompts']
    prompt = f'Extract the following structured data for {material_type} materials from the user\'s paper:\n\n'
    for field in search_fields:
        desc = search_fields[field]['prompt']
        prompt += f'- {desc}\n'
    prompt += f'\nThe output should be in JSON format, with each field as a key and the extracted data as a value. Create a new JSON string for each additional {material_type} material found. Be critical of your decisions and review your answers to make sure they reflect the information in the paper. If there are multiple materials or permutations of materials, make sure to scrape all of them. If a field is not mentioned in the paper, set the value to "None". Only put values in square brackets if multiple values are matched with the field. {additional_prompts} For example:\n\n'
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
    response = gpt_query(messages, 'gpt-5-mini').replace('\n', '')
    materials = re.findall(r'\{.*?\}', response)
    data = []
    for material_json in materials:
        material = json.loads(material_json)
        data.append(material)
    return data

def gpt_pdf_scrape(filepath, recipe):
    material_type = recipe['material type']
    search_fields = recipe['search fields']
    additional_prompts = recipe['additional prompts']
    
    client = openai.OpenAI()
    
    # Upload the file to OpenAI
    with open(filepath, 'rb') as file:
        uploaded_file = client.files.create(file=file, purpose='assistants')
    
    file_id = uploaded_file.id
    
    # Create an assistant
    assistant = client.beta.assistants.create(
        name='Material Data Extractor',
        instructions=f'Extract the following structured data for {material_type} materials from the user\'s paper:\n\n',
        model="gpt-5-mini"
    )
    
    assistant_id = assistant.id
    
    # Create a thread
    thread = client.beta.threads.create()
    thread_id = thread.id
    
    # Create a message in the thread
    message_content = "Extract material data based on the following fields: "
    for field, details in search_fields.items():
        message_content += f"{field}: {details['prompt']}\n"
    message_content += additional_prompts
    
    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=message_content,
        file_ids=[file_id]
    )
    
    # Run the assistant
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id
    )
    
    # Wait for completion
    while run.status not in ["completed", "failed"]:
        run = client.beta.threads.runs.retrieve(run.id)
    
    if run.status == "failed":
        raise RuntimeError("Assistant processing failed")
    
    # Retrieve the response
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    response = messages.data[0].content[0].text.value.replace('\n', '')
    
    materials = re.findall(r'\{.*?\}', response)
    data = []
    
    for material_json in materials:
        material = json.loads(material_json)
        data.append(material)
    
    return data

def gpt_unit_conversion(values, field, unit):
    prompt = f'Convert the following values of {field} to {unit}. Each result should be returned as a decimal on a separate line. If the input contains multiple values on one line, return the converted values as a python list on the same line. Only put values in square brackets if multiple values are provided on the line. Do not include the units. If you are unsure how to do the conversion, just return the original value. If a range is given, report this as two decimals with a hyphen/dash inbetween (For example: 1-10). If the value is already in the desired unit, just convert it to a decimal. Do not return "None". Do not return the value as an addition. If text is given and cannot be meaningfully converted, return the same text. Convert "RT" or "Room temperature" to the equivalent of 298.15K. Do not use quotation marks. Make sure that there are as many output values as input.'
    values_str = ''
    memory = []
    for value in values:
        if str(value) in ['nan',"['None']"]:
            memory.append(0)
        else:
            memory.append(1)
            value = str(value)
            values_str += f'{value}\n'
    coeff = token_length(values_str, 'gpt-5-mini')/200000
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
    temp = []
    if values_strs[0] != '':
        for values_str in values_strs:
            messages=[
                {
                    'role': 'system',
                    'content': prompt
                },
                {'role': 'user', 'content': values_str},
            ]
            converted_values = gpt_query(messages,'gpt-5-mini').splitlines()
            for value in converted_values:
                temp.append(value)
    index = 0
    for mem in memory:
        if mem == 0:
            output.append(None)
        elif mem == 1:
            output.append(temp[index])
            index+=1
    return output