"""Build extraction prompts, call models, parse JSON, and convert units.

This module turns paper text or images into structured material records using a
recipe schema. It also reconciles text/image results and normalizes extracted
units before records are stored.
"""

import json
import math
import re
from json import JSONDecodeError

import tiktoken

from paperscraper.models import ModelConfig, query_images, query_text
from paperscraper.settings import DEFAULT_MODEL

def token_length(prompt, model=DEFAULT_MODEL):
    """Estimate token length for text, falling back to a character heuristic."""
    if type(prompt) != str:
        return []
    try:
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(prompt))
    except Exception:
        return max(1, math.ceil(len(prompt) / 4))


def _field_schema(recipe):
    """Render recipe fields as a prompt-readable schema list."""
    lines = []
    for field, config in recipe['search fields'].items():
        lines.append(f'- "{field}": {config["prompt"]}')
    return '\n'.join(lines)


def _example_record(recipe):
    """Build a JSON example record from recipe example values."""
    record = {}
    for field, config in recipe['search fields'].items():
        record[field] = config['example']
    return json.dumps([record], indent=2)


def _base_extraction_prompt(recipe, source, source_rules):
    """Build the shared extraction prompt for text or image sources."""
    material_type = recipe['material type']
    additional_prompts = recipe.get('additional prompts', '')
    return f'''You extract structured {material_type} materials data from a scientific {source}.

Extraction rules:
- Use only information supported by the provided {source}. Do not infer missing values from general domain knowledge.
- Prefer explicit reported values over derived or assumed values. Preserve conditions such as temperature, pressure, composition, measurement method, and units when they are part of the reported value.
- If multiple distinct {material_type} materials, doped variants, compositions, or experimental entries are reported, return one record per distinct entry. Do not duplicate records for repeated mentions of the same entry.
- If a field is not supported by the provided {source}, set that field to "None".
- Use lists only when multiple values are reported for the same field in the same record.
- Keep values concise but complete enough to preserve scientific meaning.
{source_rules}

Schema. Use these keys exactly and do not add extra keys:
{_field_schema(recipe)}

Additional recipe instructions:
{additional_prompts}

Output contract:
- Return a JSON array of objects. Return [] if no relevant material data is present.
- Every object must contain every schema key exactly once.
- Return only JSON. Do not include markdown fences, comments, explanations, or prose.

Example output shape:
{_example_record(recipe)}'''


def build_text_extraction_prompt(recipe):
    """Build the prompt used for extraction from paper text."""
    source_rules = '''- Treat abstracts, captions, tables, experimental sections, results, and supporting text as valid evidence.
- Do not use references, citations, or background discussion as evidence for the paper's own reported measurements unless the text clearly states the value belongs to this work.'''
    return _base_extraction_prompt(recipe, 'paper text', source_rules)


def build_image_extraction_prompt(recipe, with_context=False):
    """Build the prompt used for extraction from PDF images or rendered pages."""
    context_rule = '- You may use the supplied paper text as context, but values should still be tied to the image, caption, table, plot labels, legend, or nearby visual evidence.' if with_context else '- Use only information visible in the supplied image or images, including captions, tables, plot labels, axes, legends, annotations, and readable text.'
    source_rules = f'''{context_rule}
- Actively inspect figures and tables in the image for the requested properties, including captions, table headers, table rows, plot axes, legends, labels, annotations, and inset text.
- For plots, report values only when they can be read from labels, annotations, tables, or clearly interpretable plotted data.
- If several images are supplied together, combine evidence across them only when they clearly describe the same material or experiment.
- If the supplied image or images are decorative, unreadable, or irrelevant to the schema, return [].'''
    return _base_extraction_prompt(recipe, 'paper image', source_rules)


def build_scrape_prompt(recipe, source='paper', with_context=False):
    """Select the appropriate extraction prompt for a text or image source."""
    if source == 'paper image':
        return build_image_extraction_prompt(recipe, with_context=with_context)
    return build_text_extraction_prompt(recipe)


def query_model(messages, model_config=None):
    """Send extraction messages to the configured text model."""
    config = model_config or ModelConfig.from_profile('text')
    return query_text(messages, config=config, max_output_tokens=10000)


def _strip_fences(response):
    """Remove surrounding Markdown code fences from a model response."""
    response = response.strip()
    fence = chr(96) * 3
    if response.startswith(fence):
        response = re.sub(r'^' + fence + r'(?:json)?', '', response, flags=re.IGNORECASE).strip()
        response = re.sub(fence + r'$', '', response).strip()
    return response


def _json_decoder_scan(response):
    """Scan a messy response for JSON objects or arrays."""
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
    """Parse model output into a list of JSON object records."""
    response = _strip_fences(response)
    try:
        parsed = json.loads(response)
    except JSONDecodeError:
        parsed = None
    if parsed == []:
        return []
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if isinstance(parsed, dict):
        return [parsed]

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
    """Extract structured material records from paper text."""
    prompt = build_scrape_prompt(recipe, source='paper')
    messages = [
        {'role': 'system', 'content': prompt},
        {'role': 'user', 'content': text},
    ]
    response = query_model(messages, model_config=model_config)
    return _extract_json_objects(response)


def scrape_images(image_paths, recipe, model_config=None, context=None):
    """Extract structured material records from one or more paper images."""
    config = model_config or ModelConfig.from_profile('vision')
    prompt = build_scrape_prompt(recipe, source='paper image', with_context=context is not None)
    response = query_images(prompt, image_paths, config=config, context=context, max_output_tokens=10000)
    return _extract_json_objects(response)


def combine_material_records(text_materials, image_materials, recipe, model_config=None):
    """Ask the text model to reconcile text-derived and image-derived records."""
    prompt = f'''You reconcile structured materials data extracted from the same paper.

The text extractor and image extractor may describe the same material, composition, sample, or experiment. Compare both sets and merge records that refer to the same material or clearly corresponding experimental entry.

Rules:
- Use the recipe schema keys exactly. Do not add extra keys.
- Match records first by exact material name, formula, composition, stoichiometry, dopant, substitution level, composite/additive identity, and sample label when available.
- Treat records as the same material only when their composition and experimental context are compatible. If one record is generic and another is specific, keep the specific record and copy generic fields only when they clearly apply.
- Do not merge entries when compositions differ, dopant levels differ, one is a parent phase and the other is a doped/substituted phase, or one is a neat material and the other is a composite/additive mixture.
- Do not merge entries only because they share a property value, measurement type, figure/table, or broad material class.
- Prefer explicit non-None values over None.
- If text and image records provide complementary properties for the same material, combine them into one record.
- If text and image records conflict, keep the value that is more specific or better supported; if the conflict cannot be resolved, keep both values as a list.
- Keep genuinely distinct materials, doped variants, compositions, samples, or experimental entries as separate records.
- Do not invent values. If neither source supports a field, use "None".
- Return a JSON array of reconciled records only. Do not include markdown, comments, or prose.

Schema. Use these keys exactly and do not add extra keys:
{_field_schema(recipe)}

Example output shape:
{_example_record(recipe)}'''
    payload = {
        'text_extracted_records': text_materials,
        'image_extracted_records': image_materials,
    }
    messages = [
        {'role': 'system', 'content': prompt},
        {'role': 'user', 'content': json.dumps(payload, indent=2)},
    ]
    response = query_model(messages, model_config=model_config)
    return _extract_json_objects(response)


def scrape_pdf(filepath, recipe, model_config=None):
    """Extract material records from text read directly from a PDF file."""
    from paperscraper.documents import read_pdf_text

    return scrape_text(read_pdf_text(filepath), recipe, model_config=model_config)


def convert_units(values, field, unit, model_config=None):
    """Use the text model to convert extracted values into a target unit."""
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
