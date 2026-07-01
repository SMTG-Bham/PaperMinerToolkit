import json

import pytest

import paperscraper.extract as extract


def sample_recipe():
    return {
        'material type': 'solid electrolyte',
        'additional prompts': 'Capture only room-temperature measurements.',
        'search fields': {
            'Name': {
                'prompt': 'The material name or formula.',
                'example': 'LLZO',
            },
            'Conductivity': {
                'prompt': 'The ionic conductivity with units.',
                'example': '1e-3 S cm^-1',
            },
        },
    }


def test_token_length_handles_non_strings_model_encodings_and_fallbacks(monkeypatch):
    """
    Test token length estimation for normal and fallback paths.

    This function performs the following steps:
    1. Checks that non-string prompts return the historical empty-list value.
    2. Replaces provider-aware token counting with a deterministic fake.
    3. Calls `token_length` with both a model name and a model config.

    Asserts:
        - Non-string prompts return an empty list.
        - String prompts delegate to the provider-aware token counter.
    """

    assert extract.token_length(None) == []
    monkeypatch.setattr(extract, 'count_text_tokens', lambda prompt, model_config=None, model=None, provider=None: 3)
    assert extract.token_length('one two three', model='test-model') == 3
    assert extract.token_length('one two three', model_config={'provider': 'test'}) == 3


def test_prompt_builders_include_recipe_schema_examples_and_source_rules():
    """
    Test prompt construction from a recipe.

    This function performs the following steps:
    1. Builds text and image prompts from a sample recipe.
    2. Builds the image prompt with and without text context.
    3. Selects prompts through the public `build_scrape_prompt` helper.

    Asserts:
        - Recipe fields, examples, and additional instructions are present.
        - Text prompts include text-specific evidence rules.
        - Image prompts switch their context instructions when context is requested.
        - `build_scrape_prompt` selects the expected prompt family.
    """
    recipe = sample_recipe()

    text_prompt = extract.build_text_extraction_prompt(recipe)
    image_prompt = extract.build_image_extraction_prompt(recipe)
    contextual_image_prompt = extract.build_image_extraction_prompt(recipe, with_context=True)

    assert '"Name": The material name or formula.' in text_prompt
    assert '"Conductivity": The ionic conductivity with units.' in text_prompt
    assert 'Capture only room-temperature measurements.' in text_prompt
    assert json.dumps([{'Name': 'LLZO', 'Conductivity': '1e-3 S cm^-1'}], indent=2) in text_prompt
    assert 'Do not use references' in text_prompt
    assert 'Use only information visible' in image_prompt
    assert 'supplied paper text as context' in contextual_image_prompt
    assert extract.build_scrape_prompt(recipe, source='paper') == text_prompt
    assert extract.build_scrape_prompt(recipe, source='paper image') == image_prompt


def test_query_model_uses_text_profile_and_requested_output_limit(monkeypatch):
    """
    Test model query configuration for text extraction.

    This function performs the following steps:
    1. Replaces the text model profile loader with a local fake.
    2. Replaces the text query function with a local fake that records inputs.
    3. Calls `query_model` without an explicit model configuration.

    Asserts:
        - The text model profile is loaded.
        - Messages are passed through unchanged.
        - The output token limit requested by extraction is used.
    """
    calls = {}

    class FakeConfig:
        name = 'fake-text-model'

        @classmethod
        def from_profile(cls, profile):
            calls['profile'] = profile
            return cls()

    def fake_query_text(messages, config, max_output_tokens):
        calls['messages'] = messages
        calls['config'] = config
        calls['max_output_tokens'] = max_output_tokens
        return '[]'

    monkeypatch.setattr(extract, 'ModelConfig', FakeConfig)
    monkeypatch.setattr(extract, 'query_text', fake_query_text)

    messages = [{'role': 'user', 'content': 'extract this'}]
    assert extract.query_model(messages) == '[]'
    assert calls['profile'] == 'text'
    assert calls['messages'] == messages
    assert isinstance(calls['config'], FakeConfig)
    assert calls['max_output_tokens'] == 10000


def test_extract_json_objects_parses_clean_fenced_messy_and_invalid_responses():
    """
    Test JSON extraction from several model response shapes.

    This function performs the following steps:
    1. Parses a normal JSON array and a JSON object.
    2. Parses a fenced JSON response.
    3. Parses a response containing prose wrapped around JSON snippets.
    4. Parses a response that only matches the compact object fallback.
    5. Attempts to parse a response without JSON objects.

    Asserts:
        - JSON arrays and objects are returned as record lists.
        - Non-object list entries are ignored.
        - Fenced and messy model responses are recovered.
        - A helpful `ValueError` is raised when no JSON can be found.
    """
    assert extract._extract_json_objects('[{"Name": "LLZO"}, "skip"]') == [{'Name': 'LLZO'}]
    assert extract._extract_json_objects('{"Name": "LATP"}') == [{'Name': 'LATP'}]
    assert extract._extract_json_objects('```json\n[{"Name": "LPS"}]\n```') == [{'Name': 'LPS'}]
    assert extract._extract_json_objects('prefix [{"Name": "LGPS"}] suffix {"Name": "NASICON"}') == [
        {'Name': 'LGPS'},
        {'Name': 'NASICON'},
    ]
    assert extract._extract_json_objects('bad {not json} then\n{"Name": "LLTO"}') == [{'Name': 'LLTO'}]
    assert extract._extract_json_objects('[]') == []

    with pytest.raises(ValueError, match='Model response did not contain valid JSON objects'):
        extract._extract_json_objects('No records were found.')


def test_extract_json_objects_uses_regex_fallback_after_decoder_scan_fails(monkeypatch):
    """
    Test the final regex-based JSON extraction fallback.

    This function performs the following steps:
    1. Replaces the decoder scanner with a fake that finds no objects.
    2. Parses a response containing one malformed object and one valid flat object.
    3. Collects the records returned by the fallback parser.

    Asserts:
        - Malformed object fragments are ignored.
        - Valid flat JSON objects are returned when the decoder scan finds nothing.
    """
    monkeypatch.setattr(extract, '_json_decoder_scan', lambda _: [])

    assert extract._extract_json_objects('noise {not json} then {"Name": "LLZO"}') == [{'Name': 'LLZO'}]


def test_scrape_text_images_and_pdf_delegate_to_model_and_document_helpers(monkeypatch):
    """
    Test extraction helpers that call document and model layers.

    This function performs the following steps:
    1. Replaces text querying with a fake response for paper text extraction.
    2. Replaces vision profile loading and image querying with local fakes.
    3. Replaces PDF text reading with a local fake for PDF extraction.

    Asserts:
        - Text extraction sends a system prompt and the supplied text.
        - Image extraction requests the vision profile and passes image context through.
        - PDF extraction reads text from the requested PDF before scraping.
    """
    recipe = sample_recipe()
    calls = {}

    def fake_query_model(messages, model_config=None):
        calls.setdefault('text_messages', []).append(messages)
        calls['text_config'] = model_config
        return '[{"Name": "LLZO"}]'

    class FakeConfig:
        @classmethod
        def from_profile(cls, profile):
            calls['vision_profile'] = profile
            return {'profile': profile}

    def fake_query_images(prompt, image_paths, config, context, max_output_tokens):
        calls['image_prompt'] = prompt
        calls['image_paths'] = image_paths
        calls['image_config'] = config
        calls['image_context'] = context
        calls['image_tokens'] = max_output_tokens
        return '[{"Name": "image LLZO"}]'

    monkeypatch.setattr(extract, 'query_model', fake_query_model)
    monkeypatch.setattr(extract, 'ModelConfig', FakeConfig)
    monkeypatch.setattr(extract, 'query_images', fake_query_images)

    assert extract.scrape_text('paper body', recipe, model_config={'provider': 'test'}) == [{'Name': 'LLZO'}]
    assert calls['text_messages'][0][0]['role'] == 'system'
    assert calls['text_messages'][0][1] == {'role': 'user', 'content': 'paper body'}
    assert calls['text_config'] == {'provider': 'test'}

    assert extract.scrape_images(['figure.png'], recipe, context='nearby text') == [{'Name': 'image LLZO'}]
    assert calls['vision_profile'] == 'vision'
    assert calls['image_paths'] == ['figure.png']
    assert calls['image_config'] == {'profile': 'vision'}
    assert calls['image_context'] == 'nearby text'
    assert calls['image_tokens'] == 10000
    assert 'supplied paper text as context' in calls['image_prompt']

    import paperscraper.documents as documents

    monkeypatch.setattr(documents, 'read_pdf_text', lambda filepath: f'text from {filepath}')
    assert extract.scrape_pdf('paper.pdf', recipe) == [{'Name': 'LLZO'}]
    assert calls['text_messages'][-1][1]['content'] == 'text from paper.pdf'


def test_combine_material_records_sends_both_record_sets_for_reconciliation(monkeypatch):
    """
    Test reconciliation of text and image material records.

    This function performs the following steps:
    1. Replaces the model query helper with a fake response.
    2. Calls `combine_material_records` with text and image records.
    3. Decodes the user payload sent to the fake model.

    Asserts:
        - Both record sets are included in the reconciliation payload.
        - The system prompt includes the recipe schema.
        - Parsed model output is returned as material records.
    """
    calls = {}

    def fake_query_model(messages, model_config=None):
        calls['messages'] = messages
        calls['model_config'] = model_config
        return '[{"Name": "merged", "Conductivity": "1e-3 S cm^-1"}]'

    monkeypatch.setattr(extract, 'query_model', fake_query_model)

    output = extract.combine_material_records(
        [{'Name': 'LLZO'}],
        [{'Conductivity': '1e-3 S cm^-1'}],
        sample_recipe(),
        model_config={'provider': 'test'},
    )

    payload = json.loads(calls['messages'][1]['content'])
    assert payload == {
        'text_extracted_records': [{'Name': 'LLZO'}],
        'image_extracted_records': [{'Conductivity': '1e-3 S cm^-1'}],
    }
    assert '"Name": The material name or formula.' in calls['messages'][0]['content']
    assert calls['model_config'] == {'provider': 'test'}
    assert output == [{'Name': 'merged', 'Conductivity': '1e-3 S cm^-1'}]


def test_convert_units_preserves_missing_values_and_queries_only_real_values(monkeypatch):
    """
    Test unit conversion model calls and missing-value reinsertion.

    This function performs the following steps:
    1. Replaces the model profile loader with a local fake configuration.
    2. Replaces token counting so the values fit in one model call.
    3. Replaces the model query helper with deterministic converted values.
    4. Converts a list containing real values and missing placeholders.

    Asserts:
        - Missing placeholders are returned as `None`.
        - Only real values are sent to the model for conversion.
        - Converted values are reinserted in the original order.
    """
    calls = {}

    class FakeConfig:
        name = 'fake-text-model'

        @classmethod
        def from_profile(cls, profile):
            calls['profile'] = profile
            return cls()

    def fake_query_model(messages, model_config=None):
        calls['messages'] = messages
        calls['model_config'] = model_config
        return '0.001\n0.002'

    monkeypatch.setattr(extract, 'ModelConfig', FakeConfig)
    monkeypatch.setattr(extract, 'token_length', lambda prompt, model_config=None, model=None, provider=None: 1)
    monkeypatch.setattr(extract, 'query_model', fake_query_model)

    values = ['1e-3 S cm^-1', 'nan', "['None']", '2e-3 S cm^-1']
    output = extract.convert_units(values, 'Conductivity', 'S cm^-1')

    assert output == ['0.001', None, None, '0.002']
    assert calls['profile'] == 'text'
    assert calls['model_config'].name == 'fake-text-model'
    assert calls['messages'][1]['content'] == '1e-3 S cm^-1\n2e-3 S cm^-1\n'
    assert 'Convert the following values of Conductivity to S cm^-1' in calls['messages'][0]['content']


def test_convert_units_splits_large_value_batches_without_cutting_lines(monkeypatch):
    """
    Test large unit conversion batches are split between complete input values.

    This function performs the following steps:
    1. Supplies enough token length to force chunked conversion.
    2. Replaces the model query helper with a fake that records each chunk.
    3. Converts three input values.

    Asserts:
        - More than one model call is made.
        - Each chunk contains complete newline-delimited values.
        - Converted values are returned in their original order.
    """

    class FakeConfig:
        name = 'fake-text-model'

        @classmethod
        def from_profile(cls, _):
            return cls()

    chunks = []
    reserve_calls = []

    def fake_query_model(messages, model_config=None):
        chunk = messages[1]['content']
        chunks.append(chunk)
        return '\n'.join(f'converted {value}' for value in chunk.splitlines())

    def fake_reserve(prompt, model_config=None, buffer_tokens=500):
        reserve_calls.append({
            'prompt': prompt,
            'model_config': model_config,
            'buffer_tokens': buffer_tokens,
        })
        return 500

    monkeypatch.setattr(extract, 'ModelConfig', FakeConfig)
    monkeypatch.setattr(extract, 'prompt_token_reserve', fake_reserve)
    monkeypatch.setattr(extract, 'usable_input_token_limit', lambda model_config=None, reserve_tokens=0: 200000)
    monkeypatch.setattr(extract, 'token_length', lambda prompt, model_config=None, model=None, provider=None: 600000)
    monkeypatch.setattr(extract, 'query_model', fake_query_model)

    output = extract.convert_units(['alpha', 'beta', 'gamma'], 'Conductivity', 'S cm^-1')

    assert len(chunks) > 1
    assert reserve_calls[0]['model_config'].name == 'fake-text-model'
    assert reserve_calls[0]['buffer_tokens'] == 500
    assert 'Convert the following values of Conductivity to S cm^-1' in reserve_calls[0]['prompt']
    assert ''.join(chunks) == 'alpha\nbeta\ngamma\n'
    assert all(chunk.endswith('\n') for chunk in chunks)
    assert output == ['converted alpha', 'converted beta', 'converted gamma']


def test_convert_units_returns_missing_values_without_calling_model(monkeypatch):
    """
    Test unit conversion when all values are missing.

    This function performs the following steps:
    1. Replaces the model profile loader with a local fake configuration.
    2. Replaces the model query helper with a function that fails if called.
    3. Converts a list containing only missing placeholders.

    Asserts:
        - A list of `None` values is returned.
        - No model query is made when no real values are present.
    """

    class FakeConfig:
        name = 'fake-text-model'

        @classmethod
        def from_profile(cls, _):
            return cls()

    def fail_query_model(*_, **__):
        raise AssertionError('query_model should not be called')

    monkeypatch.setattr(extract, 'ModelConfig', FakeConfig)
    monkeypatch.setattr(extract, 'query_model', fail_query_model)

    assert extract.convert_units(['nan', "['None']"], 'Conductivity', 'S cm^-1') == [None, None]
