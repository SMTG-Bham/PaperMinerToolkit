"""Build extraction prompts, call models, parse JSON, and convert units.

This module turns paper text or images into structured recipe-defined records.
It also reconciles text/image results and normalizes extracted units before
records are stored.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
import math
from os import PathLike
import re
from json import JSONDecodeError
from typing import Any

from paperminer.extraction.compression import CompressionConfig
from paperminer.extraction.models import ModelConfig, query_images, query_text
from paperminer.settings import DEFAULT_MODEL
from paperminer.extraction.tokenizer import _ModelConfigSource, count_text_tokens, prompt_token_reserve, usable_input_token_limit


def token_length(
    prompt: object,
    model: str = DEFAULT_MODEL,
    model_config: _ModelConfigSource | None = None,
    provider: str | None = None,
) -> int | list[object]:
    """Estimate the token length of a prompt.

    Parameters
    ----------
    prompt : object
        Text to measure. Non-string values return an empty list for backward
        compatibility.
    model : str, optional
        Model name used for tokenizer selection.
    model_config : _ModelConfigSource or None, optional
        Model configuration used for provider-aware token counting.
    provider : str | None, optional
        Provider name used when ``model_config`` is not supplied.

    Returns
    -------
    int or list[object]
        Estimated token count, or an empty list for non-string input.
    """
    if type(prompt) != str:
        return []
    return count_text_tokens(prompt, model_config=model_config, model=model, provider=provider)


def _field_schema(recipe: Mapping[str, Any]) -> str:
    """Render recipe fields as a prompt-readable schema.

    Parameters
    ----------
    recipe : Mapping[str, Any]
        Extraction recipe containing ``search fields`` definitions.

    Returns
    -------
    str
        Newline-delimited schema entries.
    """
    lines = []
    for field, config in recipe['search fields'].items():
        lines.append(f'- "{field}": {config["prompt"]}')
    return '\n'.join(lines)


def _example_record(recipe: Mapping[str, Any]) -> str:
    """Build a JSON example from recipe values.

    Parameters
    ----------
    recipe : Mapping[str, Any]
        Extraction recipe containing field examples.

    Returns
    -------
    str
        Pretty-printed JSON array containing one example record.
    """
    record = {}
    for field, config in recipe['search fields'].items():
        record[field] = config['example']
    return json.dumps([record], indent=2)


def _base_extraction_prompt(recipe: Mapping[str, Any], source: str, source_rules: str) -> str:
    """Build the shared extraction prompt for a source type.

    Parameters
    ----------
    recipe : Mapping[str, Any]
        Extraction recipe defining the record type and fields.
    source : str
        Human-readable source description for the prompt.
    source_rules : str
        Source-specific extraction instructions.

    Returns
    -------
    str
        Complete model system prompt.
    """
    definition = recipe['record definition']
    subject = definition['subject']
    singular = definition['singular']
    plural = definition['plural']
    unit = definition['unit']
    additional_prompts = recipe.get('additional prompts', '')
    return f'''You extract structured records about {subject} from a scientific {source}.

Record definition:
- Each output object represents {unit}.
- The recipe terminology is "{singular}" for one record subject and "{plural}" for multiple record subjects.

Extraction rules:
- Use only information supported by the provided {source}. Do not infer missing values from general domain knowledge.
- Prefer explicit reported values over derived or assumed values. Preserve relevant conditions, methods, units, qualifiers, and uncertainties when they are part of the reported value.
- If multiple distinct {plural} are reported, return one record per {singular} according to the record definition. Do not duplicate records for repeated mentions of the same {singular}.
- If a field is not supported by the provided {source}, set that field to "None".
- Use lists only when multiple values are reported for the same field in the same record, unless the field or additional recipe instructions explicitly require a list.
- Keep values concise but complete enough to preserve scientific meaning.
{source_rules}

Schema. Use these keys exactly and do not add extra keys:
{_field_schema(recipe)}

Additional recipe instructions:
{additional_prompts}

Output contract:
- Return a JSON array of objects. Return [] if no relevant {plural} are present.
- Every object must contain every schema key exactly once.
- Return only JSON. Do not include markdown fences, comments, explanations, or prose.

Example output shape:
{_example_record(recipe)}'''


def build_text_extraction_prompt(recipe: Mapping[str, Any]) -> str:
    """Build an extraction prompt for paper text.

    Parameters
    ----------
    recipe : Mapping[str, Any]
        Extraction recipe defining the requested output.

    Returns
    -------
    str
        Text extraction system prompt.
    """
    source_rules = '''- Treat abstracts, captions, tables, experimental sections, results, and supporting text as valid evidence.
- Do not use references, citations, or background discussion as evidence for the paper's own results or observations unless the text clearly states the information belongs to this work.'''
    return _base_extraction_prompt(recipe, 'paper text', source_rules)


def build_image_extraction_prompt(recipe: Mapping[str, Any], with_context: bool = False) -> str:
    """Build an extraction prompt for paper images.

    Parameters
    ----------
    recipe : Mapping[str, Any]
        Extraction recipe defining the requested output.
    with_context : bool, optional
        Whether accompanying paper text will be provided.

    Returns
    -------
    str
        Image extraction system prompt.
    """
    context_rule = '- You may use the supplied paper text as context, but values should still be tied to the image, caption, table, plot labels, legend, or nearby visual evidence.' if with_context else '- Use only information visible in the supplied image or images, including captions, tables, plot labels, axes, legends, annotations, and readable text.'
    source_rules = f'''{context_rule}
- Actively inspect figures and tables in the image for the requested properties, including captions, table headers, table rows, plot axes, legends, labels, annotations, and inset text.
- For plots, report values only when they can be read from labels, annotations, tables, or clearly interpretable plotted data.
- If several images are supplied together, combine evidence across them only when they clearly describe the same record subject or observation.
- If the supplied image or images are decorative, unreadable, or irrelevant to the schema, return [].'''
    return _base_extraction_prompt(recipe, 'paper image', source_rules)


def build_scrape_prompt(
    recipe: Mapping[str, Any],
    source: str = 'text',
    with_context: bool = False,
) -> str:
    """Select an extraction prompt for a source type.

    Parameters
    ----------
    recipe : Mapping[str, Any]
        Extraction recipe defining the requested output.
    source : str, optional
        Source type. ``"image"`` selects the image prompt; all other values
        select the text prompt.
    with_context : bool, optional
        Whether image extraction will include paper text context.

    Returns
    -------
    str
        Source-specific extraction prompt.
    """
    if source == 'image':
        return build_image_extraction_prompt(recipe, with_context=with_context)
    return build_text_extraction_prompt(recipe)


def query_model(messages: list[dict[str, Any]], model_config: ModelConfig | None = None) -> str:
    """Send extraction messages to a text model.

    Parameters
    ----------
    messages : list[dict[str, Any]]
        Provider-neutral chat messages.
    model_config : ModelConfig | None, optional
        Model configuration. The text profile is used by default.

    Returns
    -------
    str
        Model response text.
    """
    config = model_config or ModelConfig.from_profile('text')
    return query_text(messages, config=config, max_output_tokens=10000)


def _strip_fences(response: str) -> str:
    """Remove surrounding Markdown code fences from a response.

    Parameters
    ----------
    response : str
        Raw model response.

    Returns
    -------
    str
        Response without an outer Markdown code fence.
    """
    response = response.strip()
    fence = chr(96) * 3
    if response.startswith(fence):
        response = re.sub(r'^' + fence + r'(?:json)?', '', response, flags=re.IGNORECASE).strip()
        response = re.sub(fence + r'$', '', response).strip()
    return response


def _json_decoder_scan(response: str) -> list[dict[str, Any]]:
    """Scan a mixed response for JSON objects and arrays.

    Parameters
    ----------
    response : str
        Model response that may contain prose around JSON values.

    Returns
    -------
    list[dict[str, Any]]
        Object records decoded from the response.
    """
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


def _extract_json_objects(response: str) -> list[dict[str, Any]]:
    """Parse model output into JSON object records.

    Parameters
    ----------
    response : str
        Raw model response.

    Returns
    -------
    list[dict[str, Any]]
        Parsed object records.

    Raises
    ------
    ValueError
        If the response contains no valid JSON objects.
    """
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


def scrape_text(
    text: str,
    recipe: Mapping[str, Any],
    model_config: ModelConfig | None = None,
) -> list[dict[str, Any]]:
    """Extract structured recipe-defined records from paper text.

    Parameters
    ----------
    text : str
        Paper text to analyze.
    recipe : Mapping[str, Any]
        Extraction recipe defining the requested records.
    model_config : ModelConfig | None, optional
        Text model configuration.

    Returns
    -------
    list[dict[str, Any]]
        Extracted records.
    """
    prompt = build_scrape_prompt(recipe, source='text')
    messages = [
        {'role': 'system', 'content': prompt},
        {'role': 'user', 'content': text},
    ]
    response = query_model(messages, model_config=model_config)
    return _extract_json_objects(response)


def scrape_images(
    image_paths: list[str],
    recipe: Mapping[str, Any],
    model_config: ModelConfig | None = None,
    context: str | None = None,
    compression_config: CompressionConfig | None = None,
) -> list[dict[str, Any]]:
    """Extract structured recipe-defined records from paper images.

    Parameters
    ----------
    image_paths : list[str]
        Local paths to images to analyze together.
    recipe : Mapping[str, Any]
        Extraction recipe defining the requested records.
    model_config : ModelConfig | None, optional
        Vision model configuration.
    context : str | None, optional
        Paper text supplied as additional context.
    compression_config : CompressionConfig | None, optional
        Compression policy for the model payload.

    Returns
    -------
    list[dict[str, Any]]
        Extracted records.
    """
    config = model_config or ModelConfig.from_profile('vision')
    prompt = build_scrape_prompt(recipe, source='image', with_context=context is not None)
    response = query_images(prompt,
                            image_paths,
                            config=config,
                            context=context,
                            max_output_tokens=10000,
                            compression_config=compression_config)
    return _extract_json_objects(response)


def combine_material_records(
    text_materials: list[dict[str, Any]],
    image_materials: list[dict[str, Any]],
    recipe: Mapping[str, Any],
    model_config: ModelConfig | None = None,
) -> list[dict[str, Any]]:
    """Reconcile text-derived and image-derived recipe records.

    Parameters
    ----------
    text_materials : list[dict[str, Any]]
        Records extracted from paper text.
    image_materials : list[dict[str, Any]]
        Records extracted from paper images.
    recipe : Mapping[str, Any]
        Extraction recipe defining the record schema.
    model_config : ModelConfig | None, optional
        Text model configuration used for reconciliation.

    Returns
    -------
    list[dict[str, Any]]
        Reconciled records.
    """
    definition = recipe['record definition']
    subject = definition['subject']
    singular = definition['singular']
    plural = definition['plural']
    unit = definition['unit']
    identity_fields = definition['identity fields']
    additional_prompts = recipe.get('additional prompts', '')
    identity_description = ', '.join(f'"{field}"' for field in identity_fields)
    if not identity_description:
        identity_description = 'No primary identity fields are configured; use the full record context.'

    prompt = f'''You reconcile two sets of structured records about {subject} extracted from the same paper.

Each output object represents {unit}. The recipe terminology is "{singular}" for one record subject and "{plural}" for multiple record subjects.
Compare the text-derived and image-derived records and merge only records that refer to the same {singular}.

Primary identity fields:
{identity_description}

Rules:
- Use the recipe schema keys exactly. Do not add extra keys.
- Use compatible values in the primary identity fields as the strongest evidence that two records describe the same {singular}. Also consider the full record context and the configured record unit.
- If one record has a more specific identity than another, merge them only when the less-specific record clearly applies to that same {singular}.
- Keep records separate when their identity fields or defining context conflict.
- Do not merge records only because they share a reported value, method, source location, or broad category.
- Prefer explicit non-None values over None.
- If text and image records provide complementary fields for the same {singular}, combine them into one record.
- If text and image records conflict, keep the value that is more specific or better supported; if the conflict cannot be resolved, keep both values as a list.
- Keep genuinely distinct {plural} as separate records according to the configured record unit.
- Do not invent values. If neither source supports a field, use "None".
- Return a JSON array of reconciled records only. Do not include markdown, comments, or prose.

Additional recipe instructions:
{additional_prompts}

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


def scrape_pdf(
    filepath: str | PathLike[str],
    recipe: Mapping[str, Any],
    model_config: ModelConfig | None = None,
) -> list[dict[str, Any]]:
    """Extract recipe-defined records from a PDF file.

    Parameters
    ----------
    filepath : str | PathLike[str]
        Path to the source PDF.
    recipe : Mapping[str, Any]
        Extraction recipe defining the requested records.
    model_config : ModelConfig | None, optional
        Text model configuration.

    Returns
    -------
    list[dict[str, Any]]
        Extracted records.
    """
    from paperminer.corpus.documents import read_pdf_text

    return scrape_text(read_pdf_text(filepath), recipe, model_config=model_config)


def convert_units(
    values: Iterable[object],
    field: str,
    unit: str,
    model_config: ModelConfig | None = None,
) -> list[str | None]:
    """Convert extracted values into a target unit with a text model.

    Parameters
    ----------
    values : Iterable[object]
        Extracted values to convert.
    field : str
        Name of the measured field.
    unit : str
        Desired output unit.
    model_config : ModelConfig | None, optional
        Text model configuration.

    Returns
    -------
    list[str | None]
        Converted values, with missing inputs represented by ``None``.
    """
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
    reserve_tokens = prompt_token_reserve(prompt, model_config=config, buffer_tokens=500)
    token_budget = usable_input_token_limit(config, reserve_tokens=reserve_tokens)
    coeff = token_length(values_str, model_config=config) / token_budget
    if coeff <= 1:
        values_strs = [values_str]
    else:
        coeff = math.ceil(coeff)
        char_per_split = math.ceil(len(values_str) / coeff)
        values_strs = []
        current = ''
        for line in values_str.splitlines():
            line = f'{line}\n'
            if current and len(current) + len(line) > char_per_split:
                values_strs.append(current)
                current = ''
            current += line
        if current:
            values_strs.append(current)
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
