from click.testing import CliRunner

import cli.cli as cli


def test_paper_search_passes_query_path_source_and_count(monkeypatch):
    """
    Test the paper search command delegates to the search workflow.

    This function performs the following steps:
    1. Replaces `search_for_papers` with a local fake that records its inputs.
    2. Runs the CLI command with explicit query, path, source, and count values.
    3. Reads the command result and recorded call.

    Asserts:
        - The command exits successfully.
        - The search workflow receives the requested query, path, source, and count.
    """
    calls = {}
    monkeypatch.setattr(
        cli,
        'search_for_papers',
        lambda query, path, source, count: calls.update({
            'query': query,
            'path': path,
            'source': source,
            'count': count,
        }),
    )

    result = CliRunner().invoke(cli.paper_search, ['Lithium solid electrolyte', 'papers.csv', '--source', 'core',
                                                   '--count', '10'])

    assert result.exit_code == 0
    assert calls == {
        'query': 'Lithium solid electrolyte',
        'path': 'papers.csv',
        'source': 'core',
        'count': 10,
    }


def test_import_pdf_folder_passes_crossref_option(monkeypatch, tmp_path):
    """
    Test the PDF import command delegates with the expected Crossref option.

    This function performs the following steps:
    1. Creates a temporary import directory accepted by Click path validation.
    2. Replaces `import_pdfs` with a local fake that records its inputs.
    3. Runs the command with `--no-crossref`.

    Asserts:
        - The command exits successfully.
        - The import workflow receives `use_crossref=False`.
    """
    calls = {}
    papers_dir = tmp_path / 'papers'
    papers_dir.mkdir()
    monkeypatch.setattr(
        cli,
        'import_pdfs',
        lambda directory, path, use_crossref: calls.update({
            'directory': directory,
            'path': path,
            'use_crossref': use_crossref,
        }),
    )

    result = CliRunner().invoke(cli.import_pdf_folder, [str(papers_dir), 'external.csv', '--no-crossref'])

    assert result.exit_code == 0
    assert calls == {
        'directory': str(papers_dir),
        'path': 'external.csv',
        'use_crossref': False,
    }


def test_download_passes_format_and_sources(monkeypatch, tmp_path):
    """
    Test the download command delegates to the download workflow.

    This function performs the following steps:
    1. Creates a temporary papers CSV accepted by Click path validation.
    2. Replaces `download_papers` with a local fake that records its inputs.
    3. Runs the command with explicit format and repeated source options.

    Asserts:
        - The command exits successfully.
        - Download format and source choices are passed through.
    """
    calls = {}
    papers_path = tmp_path / 'papers.csv'
    papers_path.write_text('paper_id\n')
    monkeypatch.setattr(
        cli,
        'download_papers',
        lambda path, directory, download_format, sources: calls.update({
            'path': path,
            'directory': directory,
            'download_format': download_format,
            'sources': sources,
        }),
    )

    result = CliRunner().invoke(cli.download, [str(papers_path), 'papers', '--format', 'pdf', '--source', 'core',
                                               '--source', 'unpaywall'])

    assert result.exit_code == 0
    assert calls == {
        'path': str(papers_path),
        'directory': 'papers',
        'download_format': 'pdf',
        'sources': ['core', 'unpaywall'],
    }


def test_scrape_passes_model_image_cleanup_and_output_options(monkeypatch, tmp_path):
    """
    Test the scrape command delegates with text, image, and cleanup options.

    This function performs the following steps:
    1. Creates temporary paper directory and papers CSV inputs accepted by Click validation.
    2. Replaces `scrape_papers` with a local fake that records its inputs.
    3. Runs the command with explicit text, vision, image, cleanup, output, and force options.

    Asserts:
        - The command exits successfully.
        - All scrape options are forwarded to the scrape workflow.
    """
    calls = {}
    papers_dir = tmp_path / 'papers'
    papers_dir.mkdir()
    papers_path = tmp_path / 'papers.csv'
    papers_path.write_text('paper_id\n')

    def fake_scrape_papers(*args, **kwargs):
        calls['args'] = args
        calls['kwargs'] = kwargs

    monkeypatch.setattr(cli, 'scrape_papers', fake_scrape_papers)

    result = CliRunner().invoke(cli.scrape, [
        str(papers_dir),
        str(papers_path),
        'custom_recipe',
        '--mode', 'text-images',
        '--image-context', 'paper-text',
        '--image-dir', 'images',
        '--image-extraction', 'pages',
        '--image-dpi', '150',
        '--image-batch-size', 'all',
        '--model', 'text-model',
        '--provider', 'local',
        '--base-url', 'http://localhost:8000/v1',
        '--vision-model', 'vision-model',
        '--vision-provider', 'local',
        '--vision-base-url', 'http://localhost:8000/v1',
        '--delete-images-after',
        '--delete-papers-after',
        '--output', 'scraped.csv',
        '--force',
    ])

    assert result.exit_code == 0
    assert calls['args'] == (str(papers_dir), str(papers_path), 'custom_recipe')
    assert calls['kwargs'] == {
        'mode': 'text-images',
        'image_dir': 'images',
        'image_context': 'paper-text',
        'image_extraction': 'pages',
        'image_dpi': 150,
        'image_batch_size': 'all',
        'model': 'text-model',
        'provider': 'local',
        'base_url': 'http://localhost:8000/v1',
        'vision_model': 'vision-model',
        'vision_provider': 'local',
        'vision_base_url': 'http://localhost:8000/v1',
        'delete_images_after': True,
        'delete_papers_after': True,
        'output_path': 'scraped.csv',
        'force': True,
    }


def test_store_passes_files_recipe_and_assume_yes(monkeypatch, tmp_path):
    """
    Test the store command delegates to the store workflow.

    This function performs the following steps:
    1. Creates a temporary papers CSV accepted by Click path validation.
    2. Replaces `store_results` with a local fake that records its inputs.
    3. Runs the command with explicit input, output, recipe, and assume-yes options.

    Asserts:
        - The command exits successfully.
        - Store arguments include unit conversion enabled and assume-yes forwarded.
    """
    calls = {}
    papers_path = tmp_path / 'papers.csv'
    papers_path.write_text('paper_id\n')
    monkeypatch.setattr(
        cli,
        'store_results',
        lambda path, in_file, out_file, unit_conversion, recipe, assume_yes: calls.update({
            'path': path,
            'in_file': in_file,
            'out_file': out_file,
            'unit_conversion': unit_conversion,
            'recipe': recipe,
            'assume_yes': assume_yes,
        }),
    )

    result = CliRunner().invoke(cli.store, [str(papers_path), 'scraped.csv', 'materials.csv', 'sse', '--assume-yes'])

    assert result.exit_code == 0
    assert calls == {
        'path': str(papers_path),
        'in_file': 'scraped.csv',
        'out_file': 'materials.csv',
        'unit_conversion': True,
        'recipe': 'sse',
        'assume_yes': True,
    }


def test_key_update_entry_points_call_settings_helpers(monkeypatch):
    """
    Test API configuration entry points delegate to settings helpers.

    This function performs the following steps:
    1. Replaces each API-key update helper with a local fake that records its name.
    2. Calls each non-Click entry point directly.
    3. Collects the recorded calls.

    Asserts:
        - Each API-key or email entry point calls its matching settings helper exactly once.
    """
    calls = []
    monkeypatch.setattr(cli, 'update_elsevier_key', lambda: calls.append('elsevier'))
    monkeypatch.setattr(cli, 'update_core_key', lambda: calls.append('core'))
    monkeypatch.setattr(cli, 'update_unpaywall_email', lambda: calls.append('unpaywall'))
    monkeypatch.setattr(cli, 'update_openai_key', lambda: calls.append('openai'))
    monkeypatch.setattr(cli, 'update_anthropic_key', lambda: calls.append('anthropic'))

    cli.update_elsevier_api_key()
    cli.update_core_api_key()
    cli.update_unpaywall_api_email()
    cli.update_openai_api_key()
    cli.update_anthropic_api_key()

    assert calls == ['elsevier', 'core', 'unpaywall', 'openai', 'anthropic']


def test_model_config_infers_capabilities_saves_profile_and_prints_summary(monkeypatch):
    """
    Test the model configuration command with inferred capabilities.

    This function performs the following steps:
    1. Replaces capability inference and settings persistence with local fakes.
    2. Runs the model configuration command without explicit capabilities.
    3. Reads the command output and recorded calls.

    Asserts:
        - Capabilities are inferred from the profile and model name.
        - The model profile is saved with generation settings.
        - The printed summary includes provider, model, capabilities, and sampling settings.
    """
    calls = {}
    monkeypatch.setattr(cli, 'infer_model_capabilities', lambda profile, model: ['text'])
    monkeypatch.setattr(
        cli,
        'set_model_profile',
        lambda profile, provider, model, base_url, api_key, capabilities, temperature, top_p, input_token_limit: calls.update({
            'profile': profile,
            'provider': provider,
            'model': model,
            'base_url': base_url,
            'api_key': api_key,
            'capabilities': capabilities,
            'temperature': temperature,
            'top_p': top_p,
            'input_token_limit': input_token_limit,
        }),
    )

    result = CliRunner().invoke(cli.model_config, [
        'text',
        '--provider', 'openai',
        '--model', 'gpt-test',
        '--temperature', '0.2',
        '--top-p', '0.9',
        '--input-token-limit', '64000',
    ])

    assert result.exit_code == 0
    assert calls == {
        'profile': 'text',
        'provider': 'openai',
        'model': 'gpt-test',
        'base_url': None,
        'api_key': None,
        'capabilities': ['text'],
        'temperature': 0.2,
        'top_p': 0.9,
        'input_token_limit': 64000,
    }
    assert 'Updated text model profile: openai/gpt-test [text] temperature=0.2 top_p=0.9 input_token_limit=64000' in result.output


def test_model_config_uses_explicit_capabilities_without_inference(monkeypatch):
    """
    Test the model configuration command with explicit capability options.

    This function performs the following steps:
    1. Replaces capability inference with a fake that fails if called.
    2. Replaces settings persistence with a local fake that records capabilities.
    3. Runs the command with repeated `--capability` options.

    Asserts:
        - Capability inference is skipped.
        - Explicit capabilities are saved in command-line order.
    """
    calls = {}
    monkeypatch.setattr(
        cli,
        'infer_model_capabilities',
        lambda *_: (_ for _ in ()).throw(AssertionError('capabilities should not be inferred')),
    )
    monkeypatch.setattr(
        cli,
        'set_model_profile',
        lambda profile, provider, model, base_url, api_key, capabilities, temperature, top_p, input_token_limit: calls.update({
            'capabilities': capabilities,
            'input_token_limit': input_token_limit,
        }),
    )

    result = CliRunner().invoke(cli.model_config, [
        'vision',
        '--provider', 'local',
        '--model', 'vision-model',
        '--base-url', 'http://localhost:8000/v1',
        '--api-key', 'test-key',
        '--capability', 'text',
        '--capability', 'vision',
    ])

    assert result.exit_code == 0
    assert calls['capabilities'] == ['text', 'vision']
    assert calls['input_token_limit'] == cli.DEFAULT_INPUT_TOKEN_LIMIT


def test_model_status_prints_text_and_vision_profiles(monkeypatch):
    """
    Test the model status command prints configured profiles.

    This function performs the following steps:
    1. Replaces model-profile loading with a local fake for text and vision profiles.
    2. Runs the model status command.
    3. Reads the printed output.

    Asserts:
        - Text and vision profile summaries are printed with provider, model, capabilities, and base URL.
    """
    profiles = {
        'text': {
            'provider': 'openai',
            'model': 'gpt-test',
            'capabilities': ['text'],
            'temperature': 0.0,
            'top_p': 1.0,
            'input_token_limit': 32000,
            'base_url': None,
        },
        'vision': {
            'provider': 'local',
            'model': 'vision-test',
            'capabilities': ['text', 'vision'],
            'temperature': 0.1,
            'top_p': 0.8,
            'input_token_limit': 120000,
            'base_url': 'http://localhost:8000/v1',
        },
    }
    monkeypatch.setattr(cli, 'get_model_profile', lambda profile: profiles[profile])

    result = CliRunner().invoke(cli.model_status)

    assert result.exit_code == 0
    assert 'text: openai/gpt-test capabilities=[text] temperature=0.0 top_p=1.0 input_token_limit=32000 base_url=None' in result.output
    assert 'vision: local/vision-test capabilities=[text, vision] temperature=0.1 top_p=0.8 input_token_limit=120000' in result.output


def test_update_model_config_calls_interactive_settings_helper(monkeypatch):
    """
    Test the interactive model configuration entry point.

    This function performs the following steps:
    1. Replaces the interactive settings helper with a local fake.
    2. Calls the entry point directly.
    3. Reads the recorded calls.

    Asserts:
        - The interactive model settings helper is called once.
    """
    calls = []
    monkeypatch.setattr(cli, 'update_model_settings', lambda: calls.append('model_settings'))

    cli.update_model_config()

    assert calls == ['model_settings']


def test_utility_commands_delegate_to_maintenance_helpers(monkeypatch, tmp_path):
    """
    Test reset, status, sort, and shuffle CLI commands.

    This function performs the following steps:
    1. Creates a temporary papers CSV accepted by Click path validation.
    2. Replaces maintenance helpers with local fakes that record their inputs.
    3. Runs each maintenance command.

    Asserts:
        - Each command exits successfully.
        - Reset, status, sort, and shuffle helpers receive the expected arguments.
    """
    papers_path = tmp_path / 'papers.csv'
    papers_path.write_text('paper_id\n')
    calls = []
    monkeypatch.setattr(cli, 'reset', lambda path: calls.append(('reset', path)))
    monkeypatch.setattr(cli, 'status', lambda path: calls.append(('status', path)))
    monkeypatch.setattr(cli, 'sort', lambda path, field, ascending: calls.append(('sort', path, field, ascending)))
    monkeypatch.setattr(cli, 'shuffle', lambda path: calls.append(('shuffle', path)))

    runner = CliRunner()
    assert runner.invoke(cli.reset_scraper, [str(papers_path)]).exit_code == 0
    assert runner.invoke(cli.scraper_status, [str(papers_path)]).exit_code == 0
    assert runner.invoke(cli.sort_df, [str(papers_path), 'title', '--ascending']).exit_code == 0
    assert runner.invoke(cli.shuffle_papers, [str(papers_path)]).exit_code == 0

    assert calls == [
        ('reset', str(papers_path)),
        ('status', str(papers_path)),
        ('sort', str(papers_path), 'title', True),
        ('shuffle', str(papers_path)),
    ]
