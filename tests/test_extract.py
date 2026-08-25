"""Test prompt construction, response parsing, and extraction workflows."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, NoReturn

import pytest

from paperminer.compression import CompressionConfig
import paperminer.extract as extract


def test_record_reconciliation_without_identity_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tell reconciliation to use context when no identity field exists."""
    captured: dict[str, str] = {}

    def query(messages: list[dict[str, Any]], model_config: object = None) -> str:
        """Capture the reconciliation prompt."""
        captured['prompt'] = messages[0]['content']
        return '[]'

    monkeypatch.setattr(extract, 'query_model', query)
    recipe = {'record definition': {'subject': 'results', 'singular': 'result', 'plural': 'results', 'unit': 'one result', 'identity fields': []}, 'search fields': {}}
    assert extract.combine_material_records([], [], recipe) == []
    assert 'No primary identity fields are configured' in captured['prompt']


def sample_recipe() -> dict[str, Any]:
    """Return a minimal extraction recipe for tests."""
    return {
        'record definition': {
            'subject': 'solid electrolytes',
            'singular': 'material',
            'plural': 'materials',
            'unit': 'a distinct solid-electrolyte composition or sample',
            'identity fields': ['Name'],
        },
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


def test_token_length_handles_non_strings_model_encodings_and_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test token length estimation for normal and fallback paths."""
    assert extract.token_length(None) == []
    monkeypatch.setattr(extract, 'count_text_tokens', lambda prompt, model_config=None, model=None, provider=None: 3)
    assert extract.token_length('one two three', model='test-model') == 3
    assert extract.token_length('one two three', model_config={'provider': 'test'}) == 3


def test_prompt_builders_include_recipe_schema_examples_and_source_rules() -> None:
    """Test that prompt builders include recipe and source instructions."""
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
    assert extract.build_scrape_prompt(recipe, source='text') == text_prompt
    assert extract.build_scrape_prompt(recipe, source='image') == image_prompt


def test_non_material_recipe_controls_prompt_terminology_and_granularity() -> None:
    """Test that general prompts contain only recipe-supplied domain terminology."""
    recipe = {
        'record definition': {
            'subject': 'electrochemical cycling experiments',
            'singular': 'experiment',
            'plural': 'experiments',
            'unit': 'a distinct cell and cycling protocol',
            'identity fields': ['Cell identifier', 'Protocol'],
        },
        'additional prompts': 'Keep each temperature series together.',
        'search fields': {
            'Cell identifier': {'prompt': 'Reported cell label.', 'example': 'Cell A'},
            'Protocol': {'prompt': 'Reported cycling protocol.', 'example': 'C/10'},
        },
    }

    prompt = extract.build_text_extraction_prompt(recipe)

    assert 'records about electrochemical cycling experiments' in prompt
    assert 'Each output object represents a distinct cell and cycling protocol.' in prompt
    assert 'multiple distinct experiments' in prompt
    assert 'one record per experiment' in prompt
    assert 'no relevant experiments' in prompt
    assert 'Keep each temperature series together.' in prompt
    assert 'material' not in prompt.lower()


def test_query_model_uses_text_profile_and_requested_output_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that model queries use the text profile and output limit."""
    calls = {}

    class FakeConfig:
        """Provide a minimal text model configuration."""

        name = 'fake-text-model'

        @classmethod
        def from_profile(cls, profile: str) -> FakeConfig:
            """Record the requested profile and return a fake configuration."""
            calls['profile'] = profile
            return cls()

    def fake_query_text(
        messages: list[dict[str, str]],
        config: FakeConfig,
        max_output_tokens: int,
    ) -> str:
        """Record a text query and return an empty JSON result."""
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


def test_extract_json_objects_parses_clean_fenced_messy_and_invalid_responses() -> None:
    """Test JSON extraction from clean, fenced, messy, and invalid responses."""
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


def test_extract_json_objects_uses_regex_fallback_after_decoder_scan_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test regex recovery after the JSON decoder scan finds no records."""
    monkeypatch.setattr(extract, '_json_decoder_scan', lambda _: [])

    assert extract._extract_json_objects('noise {not json} then {"Name": "LLZO"}') == [{'Name': 'LLZO'}]


def test_scrape_text_images_and_pdf_delegate_to_model_and_document_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that extraction helpers delegate to model and document layers."""
    recipe = sample_recipe()
    calls = {}

    def fake_query_model(
        messages: list[dict[str, str]],
        model_config: Mapping[str, str] | None = None,
    ) -> str:
        """Record text messages and return one material record."""
        calls.setdefault('text_messages', []).append(messages)
        calls['text_config'] = model_config
        return '[{"Name": "LLZO"}]'

    class FakeConfig:
        """Provide a minimal vision model configuration."""

        @classmethod
        def from_profile(cls, profile: str) -> dict[str, str]:
            """Record and return the requested profile."""
            calls['vision_profile'] = profile
            return {'profile': profile}

    def fake_query_images(
        prompt: str,
        image_paths: list[str],
        config: dict[str, str],
        context: str,
        max_output_tokens: int,
        compression_config: CompressionConfig | None = None,
    ) -> str:
        """Record an image query and return one material record."""
        calls['image_prompt'] = prompt
        calls['image_paths'] = image_paths
        calls['image_config'] = config
        calls['image_context'] = context
        calls['image_tokens'] = max_output_tokens
        calls['image_compression'] = compression_config
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
    assert calls['image_compression'] is None
    assert 'supplied paper text as context' in calls['image_prompt']

    import paperminer.documents as documents

    monkeypatch.setattr(documents, 'read_pdf_text', lambda filepath: f'text from {filepath}')
    assert extract.scrape_pdf('paper.pdf', recipe) == [{'Name': 'LLZO'}]
    assert calls['text_messages'][-1][1]['content'] == 'text from paper.pdf'


def test_combine_material_records_sends_both_record_sets_for_reconciliation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that reconciliation sends both extracted record sets."""
    calls = {}

    def fake_query_model(
        messages: list[dict[str, str]],
        model_config: Mapping[str, str] | None = None,
    ) -> str:
        """Record reconciliation messages and return a merged record."""
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
    assert 'Primary identity fields:\n"Name"' in calls['messages'][0]['content']
    assert 'Capture only room-temperature measurements.' in calls['messages'][0]['content']
    assert calls['model_config'] == {'provider': 'test'}
    assert output == [{'Name': 'merged', 'Conductivity': '1e-3 S cm^-1'}]


def test_non_material_reconciliation_uses_recipe_identity_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that reconciliation is generic and uses configured identity fields."""
    calls = {}
    recipe = {
        'record definition': {
            'subject': 'cycling experiments',
            'singular': 'experiment',
            'plural': 'experiments',
            'unit': 'a distinct cell and protocol',
            'identity fields': ['Cell', 'Protocol'],
        },
        'search fields': {
            'Cell': {'prompt': 'Cell label.', 'example': 'A'},
            'Protocol': {'prompt': 'Cycling protocol.', 'example': 'C/10'},
        },
    }

    def fake_query_model(messages: list[dict[str, str]], model_config: object = None) -> str:
        """Capture the generic reconciliation prompt and return no records."""
        calls['system'] = messages[0]['content']
        return '[]'

    monkeypatch.setattr(extract, 'query_model', fake_query_model)
    assert extract.combine_material_records([], [], recipe) == []

    assert 'Primary identity fields:\n"Cell", "Protocol"' in calls['system']
    assert 'same experiment' in calls['system']
    assert 'material' not in calls['system'].lower()


def test_convert_units_preserves_missing_values_and_queries_only_real_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that unit conversion preserves missing values and queries real ones."""
    calls = {}

    class FakeConfig:
        """Provide a minimal text model configuration."""

        name = 'fake-text-model'

        @classmethod
        def from_profile(cls, profile: str) -> FakeConfig:
            """Record the requested profile and return a fake configuration."""
            calls['profile'] = profile
            return cls()

    def fake_query_model(
        messages: list[dict[str, str]],
        model_config: FakeConfig | None = None,
    ) -> str:
        """Record a unit conversion query and return converted values."""
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


def test_convert_units_splits_large_value_batches_without_cutting_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that large conversion batches split only between complete values."""

    class FakeConfig:
        """Provide a minimal text model configuration."""

        name = 'fake-text-model'

        @classmethod
        def from_profile(cls, _: str) -> FakeConfig:
            """Return a fake configuration for any profile."""
            return cls()

    chunks = []
    reserve_calls = []

    def fake_query_model(
        messages: list[dict[str, str]],
        model_config: FakeConfig | None = None,
    ) -> str:
        """Record a value chunk and return deterministic conversions."""
        chunk = messages[1]['content']
        chunks.append(chunk)
        return '\n'.join(f'converted {value}' for value in chunk.splitlines())

    def fake_reserve(
        prompt: str,
        model_config: FakeConfig | None = None,
        buffer_tokens: int = 500,
    ) -> int:
        """Record prompt reservation arguments and return a fixed reserve."""
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


def test_convert_units_returns_missing_values_without_calling_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that all-missing values bypass the conversion model."""

    class FakeConfig:
        """Provide a minimal text model configuration."""

        name = 'fake-text-model'

        @classmethod
        def from_profile(cls, _: str) -> FakeConfig:
            """Return a fake configuration for any profile."""
            return cls()

    def fail_query_model(*_: object, **__: object) -> NoReturn:
        """Fail if a model query is attempted."""
        raise AssertionError('query_model should not be called')

    monkeypatch.setattr(extract, 'ModelConfig', FakeConfig)
    monkeypatch.setattr(extract, 'query_model', fail_query_model)

    assert extract.convert_units(['nan', "['None']"], 'Conductivity', 'S cm^-1') == [None, None]
