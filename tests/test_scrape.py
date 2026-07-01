import os

import pandas as pd
import pytest

import paperscraper.scrape as scrape


def sample_recipe():
    """Return a minimal recipe for scrape unit tests."""
    return {
        'material type': 'solid electrolyte',
        'search fields': {
            'Name': {'prompt': 'Material name.', 'example': 'LLZO'},
            'Conductivity': {'prompt': 'Conductivity.', 'example': '1e-3 S cm^-1'},
        },
    }


def write_papers_csv(path, rows):
    """Write a papers CSV for scrape unit tests."""
    pd.DataFrame(rows).to_csv(path)


class FakeTqdm:
    """Minimal progress-bar replacement for scrape unit tests."""

    def __init__(self, *args, **kwargs):
        self.updates = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def update(self, amount):
        self.updates += amount


class FakeModelConfig:
    """Minimal model configuration replacement for scrape unit tests."""

    calls = []
    required = []

    def __init__(self, profile, name=None, provider=None, base_url=None):
        self.profile = profile
        self.name = name or f'{profile}-model'
        self.provider = provider
        self.base_url = base_url

    @classmethod
    def from_profile(cls, profile, name=None, provider=None, base_url=None):
        cls.calls.append({
            'profile': profile,
            'name': name,
            'provider': provider,
            'base_url': base_url,
        })
        return cls(profile, name=name, provider=provider, base_url=base_url)

    def require(self, capability):
        self.required.append(capability)


def test_text_chunks_uses_model_token_estimate_to_split_long_text(monkeypatch):
    """
    Test text chunking for short and long inputs.

    This function performs the following steps:
    1. Replaces token counting with a fake value below the split threshold.
    2. Replaces token counting with a fake value above the split threshold.
    3. Chunks representative short and long text values.

    Asserts:
        - Short text is returned as one chunk.
        - Long text is split into the expected number of ordered chunks.
    """
    text_config = FakeModelConfig('text', name='model')
    reserve_calls = []

    def fake_reserve(prompt, model_config=None, buffer_tokens=500):
        reserve_calls.append({
            'prompt': prompt,
            'model_config': model_config,
            'buffer_tokens': buffer_tokens,
        })
        return 750

    def fake_limit(model_config=None, reserve_tokens=2000):
        assert reserve_tokens == 750
        return 120000

    def short_count(text, model_config=None):
        return 120000

    monkeypatch.setattr(scrape, 'prompt_token_reserve', fake_reserve)
    monkeypatch.setattr(scrape, 'usable_input_token_limit', fake_limit)
    monkeypatch.setattr(scrape, 'token_length', short_count)
    assert scrape._text_chunks('short text', text_config, prompt='extract prompt') == ['short text']
    assert reserve_calls == [{'prompt': 'extract prompt', 'model_config': text_config, 'buffer_tokens': 500}]

    monkeypatch.setattr(scrape, 'prompt_token_reserve', lambda prompt, model_config=None, buffer_tokens=500: 500)
    monkeypatch.setattr(scrape, 'usable_input_token_limit', lambda model_config=None, reserve_tokens=2000: 120000)
    monkeypatch.setattr(scrape, 'token_length', lambda text, model_config=None: 240001)
    chunks = scrape._text_chunks('abcdef', text_config)
    assert chunks == ['ab', 'cd', 'ef']


def test_material_file_and_image_helpers_handle_common_inputs(tmp_path):
    """
    Test scrape helper functions for materials, files, paths, and image batches.

    This function performs the following steps:
    1. Appends paper provenance to extracted material records.
    2. Writes material records into a new and then existing CSV.
    3. Deletes an existing file and normalizes unsafe path parts.
    4. Builds image output keys and image batches.
    5. Calls image batching with invalid sizes.

    Asserts:
        - Material rows receive paper metadata and source fields.
        - Material CSV writes and appends rows correctly.
        - File deletion is quiet for existing and missing paths.
        - Path keys and image batches are stable.
        - Invalid image batch sizes raise `ValueError`.
    """
    row = pd.Series({
        'paper_id': 'doi:10.123/example',
        'doi': '10.123/example',
        'publication_date': '2024-01-01',
    })
    materials = scrape._append_materials([{'Name': 'LLZO'}], row, 'text', 'paper.txt')
    assert materials == [{
        'Name': 'LLZO',
        'Paper id': 'doi:10.123/example',
        'doi': '10.123/example',
        'Publication date': '2024-01-01',
        'Source': 'text',
        'Source path': 'paper.txt',
    }]

    out_path = tmp_path / 'materials.csv'
    first_material, written = scrape._write_materials(materials, True, str(out_path))
    assert first_material is False
    assert written == 1
    first_material, written = scrape._write_materials([{'Name': 'LATP'}], first_material, str(out_path))
    assert written == 1
    assert pd.read_csv(out_path, index_col=0)['Name'].tolist() == ['LLZO', 'LATP']
    assert scrape._write_materials([], first_material, str(out_path)) == (False, 0)

    delete_path = tmp_path / 'delete-me.txt'
    delete_path.write_text('remove me')
    scrape._delete_file(str(delete_path))
    scrape._delete_file(str(delete_path))
    assert not delete_path.exists()

    assert scrape._safe_path_part(' ../bad DOI value!! ') == 'bad_DOI_value'
    assert scrape._safe_path_part('') == 'paper'
    assert scrape._image_key_for_row(row, '/tmp/downloaded name.pdf') == 'downloaded_name'
    assert scrape._image_key_for_row(pd.Series({'paper_id': 'core:123/ABC'}), None) == '123_ABC'
    assert scrape._image_batches(['a', 'b', 'c'], 'all') == [['a', 'b', 'c']]
    assert scrape._image_batches(['a', 'b', 'c'], 2) == [['a', 'b'], ['c']]

    with pytest.raises(ValueError, match='positive integer'):
        scrape._image_batches(['a'], 0)
    with pytest.raises(ValueError, match='positive integer'):
        scrape._image_batches(['a'], 'many')


def test_scrape_papers_rejects_invalid_modes(tmp_path):
    """
    Test scrape option validation.

    This function performs the following steps:
    1. Calls `scrape_papers` with an invalid scrape mode.
    2. Calls `scrape_papers` with an invalid image context mode.
    3. Calls `scrape_papers` with an invalid image extraction mode.

    Asserts:
        - Each invalid option raises a helpful `ValueError`.
    """
    papers_dir = tmp_path / 'papers'
    papers_dir.mkdir()
    papers_path = tmp_path / 'papers.csv'
    write_papers_csv(papers_path, [{'paper_id': 'paper-1'}])

    with pytest.raises(ValueError, match='mode must be one of'):
        scrape.scrape_papers(str(papers_dir), papers_path=str(papers_path), mode='bad')
    with pytest.raises(ValueError, match='image_context must be one of'):
        scrape.scrape_papers(str(papers_dir), papers_path=str(papers_path), image_context='bad')
    with pytest.raises(ValueError, match='image_extraction must be one of'):
        scrape.scrape_papers(str(papers_dir), papers_path=str(papers_path), image_extraction='bad')


def test_scrape_papers_text_mode_writes_materials_updates_status_and_deletes_source(tmp_path, monkeypatch, capsys):
    """
    Test text-only scraping over downloaded text files.

    This function performs the following steps:
    1. Writes a papers CSV with one pending paper and one already-scraped paper.
    2. Writes a matching text file for the pending paper.
    3. Replaces recipe loading, model config, progress bar, token chunking, text reading, and text analysis.
    4. Calls `scrape_papers` in text mode with source cleanup enabled.
    5. Reloads the output materials and papers CSV files.

    Asserts:
        - Only the pending paper is analyzed when `force=False`.
        - Extracted materials are written with text provenance.
        - Text scrape status and material counts are updated.
        - The successfully scraped source file is deleted.
        - A skipped-stage summary is printed.
    """
    papers_dir = tmp_path / 'papers'
    papers_dir.mkdir()
    text_path = papers_dir / 'paper-1.txt'
    text_path.write_text('paper text')
    papers_path = tmp_path / 'papers.csv'
    output_path = tmp_path / 'scraped.csv'
    write_papers_csv(papers_path, [
        {'paper_id': 'paper-1', 'doi': '10.1/one', 'publication_date': '2024', 'text_scrape_status': 'pending'},
        {'paper_id': 'paper-2', 'doi': '10.1/two', 'publication_date': '2025', 'text_scrape_status': 'succeeded'},
    ])
    calls = {}
    FakeModelConfig.calls = []
    monkeypatch.setattr(scrape, 'load_recipe', lambda recipe: sample_recipe())
    monkeypatch.setattr(scrape, 'ModelConfig', FakeModelConfig)
    monkeypatch.setattr(scrape, 'tqdm', FakeTqdm)
    monkeypatch.setattr(scrape, '_text_chunks', lambda text, model_config, prompt='': ['chunk one', 'chunk two'])
    monkeypatch.setattr(scrape, 'read_document_text', lambda path: f'read {os.path.basename(path)}')

    def fake_analyze_text(text, recipe, model_config=None):
        calls.setdefault('chunks', []).append(text)
        calls.setdefault('configs', []).append(model_config)
        return [{'Name': f'material from {text}'}]

    monkeypatch.setattr(scrape, 'scrape_text', fake_analyze_text)

    scrape.scrape_papers(
        str(papers_dir),
        papers_path=str(papers_path),
        output_path=str(output_path),
        mode='text',
        recipe='sse',
        model='text-model',
        provider='local',
        base_url='http://localhost:8000/v1',
        delete_papers_after=True,
    )

    materials = pd.read_csv(output_path, index_col=0)
    papers = pd.read_csv(papers_path, index_col=0)
    output = capsys.readouterr().out
    assert calls['chunks'] == ['chunk one', 'chunk two']
    assert all(config.name == 'text-model' for config in calls['configs'])
    assert materials['Name'].tolist() == ['material from chunk one', 'material from chunk two']
    assert materials['Source'].tolist() == ['text', 'text']
    assert materials['Paper id'].tolist() == ['paper-1', 'paper-1']
    assert papers.loc[0, 'text_scrape_status'] == 'succeeded'
    assert papers.loc[0, 'num_text_materials'] == 2
    assert papers.loc[1, 'text_scrape_status'] == 'succeeded'
    assert not text_path.exists()
    assert FakeModelConfig.calls == [{
        'profile': 'text',
        'name': 'text-model',
        'provider': 'local',
        'base_url': 'http://localhost:8000/v1',
    }]
    assert 'Skipped already successful stages: text=1' in output


def test_scrape_papers_records_text_failures_when_source_is_missing(tmp_path, monkeypatch, capsys):
    """
    Test text scraping failure handling.

    This function performs the following steps:
    1. Writes a papers CSV for a paper without a matching text or PDF file.
    2. Replaces recipe loading, model config, and progress bar with local fakes.
    3. Calls `scrape_papers` in text mode.
    4. Reloads the papers CSV.

    Asserts:
        - The text scrape status is marked as failed.
        - The missing-source error is recorded.
        - No materials output file is created.
        - A no-materials summary is printed.
    """
    papers_dir = tmp_path / 'papers'
    papers_dir.mkdir()
    papers_path = tmp_path / 'papers.csv'
    output_path = tmp_path / 'scraped.csv'
    write_papers_csv(papers_path, [{'paper_id': 'missing-paper'}])
    monkeypatch.setattr(scrape, 'load_recipe', lambda recipe: sample_recipe())
    monkeypatch.setattr(scrape, 'ModelConfig', FakeModelConfig)
    monkeypatch.setattr(scrape, 'tqdm', FakeTqdm)

    scrape.scrape_papers(str(papers_dir), papers_path=str(papers_path), output_path=str(output_path))

    papers = pd.read_csv(papers_path, index_col=0)
    output = capsys.readouterr().out
    assert papers.loc[0, 'text_scrape_status'] == 'failed'
    assert 'No downloaded text or PDF file found' in papers.loc[0, 'last_error']
    assert not output_path.exists()
    assert 'No new scraped material rows were written' in output


def test_scrape_papers_text_images_combines_results_and_cleans_images(tmp_path, monkeypatch):
    """
    Test combined text and image scraping.

    This function performs the following steps:
    1. Writes matching text and PDF files for a pending paper.
    2. Replaces recipe loading, model config, progress bar, document reading, image extraction, and model analysis.
    3. Calls `scrape_papers` in text-image mode with image cleanup enabled.
    4. Reloads the output materials and papers CSV files.

    Asserts:
        - Vision config is requested and checked for vision support.
        - Text and image analysis both run.
        - Reconciled text-image records are written when combining succeeds.
        - Image metadata is recorded on the papers CSV.
        - Extracted temporary image files are deleted.
    """
    papers_dir = tmp_path / 'papers'
    papers_dir.mkdir()
    text_path = papers_dir / 'paper-1.txt'
    pdf_path = papers_dir / 'paper-1.pdf'
    text_path.write_text('paper text')
    pdf_path.write_text('pdf bytes')
    papers_path = tmp_path / 'papers.csv'
    output_path = tmp_path / 'scraped.csv'
    image_root = tmp_path / 'images'
    write_papers_csv(papers_path, [{'paper_id': 'paper-1', 'doi': '10.1/one', 'publication_date': '2024'}])
    FakeModelConfig.calls = []
    FakeModelConfig.required = []
    calls = {}

    monkeypatch.setattr(scrape, 'load_recipe', lambda recipe: sample_recipe())
    monkeypatch.setattr(scrape, 'ModelConfig', FakeModelConfig)
    monkeypatch.setattr(scrape, 'tqdm', FakeTqdm)
    monkeypatch.setattr(scrape, '_text_chunks', lambda text, model_config, prompt='': ['single chunk'])
    monkeypatch.setattr(scrape, 'read_document_text', lambda path: 'text context')

    def fake_extract_pdf_images(pdf_path_arg, output_dir, prefix, strategy, dpi):
        image_paths = [os.path.join(output_dir, f'{prefix}-1.png'), os.path.join(output_dir, f'{prefix}-2.png')]
        os.makedirs(output_dir, exist_ok=True)
        for path in image_paths:
            open(path, 'w').close()
        calls['image_extract'] = {
            'pdf_path': pdf_path_arg,
            'output_dir': output_dir,
            'prefix': prefix,
            'strategy': strategy,
            'dpi': dpi,
        }
        return image_paths

    def fake_analyze_text(text, recipe, model_config=None):
        calls['text'] = text
        return [{'Name': 'text LLZO'}]

    def fake_analyze_images(image_paths, recipe, model_config=None, context=None):
        calls.setdefault('image_batches', []).append(image_paths)
        calls.setdefault('contexts', []).append(context)
        return [{'Name': f'image {len(image_paths)}'}]

    def fake_combine(text_materials, image_materials, recipe, model_config=None):
        calls['combine'] = (text_materials, image_materials, model_config.name)
        return [{'Name': 'combined LLZO'}]

    monkeypatch.setattr(scrape, 'extract_pdf_images', fake_extract_pdf_images)
    monkeypatch.setattr(scrape, 'scrape_text', fake_analyze_text)
    monkeypatch.setattr(scrape, 'scrape_images', fake_analyze_images)
    monkeypatch.setattr(scrape, 'combine_material_records', fake_combine)

    scrape.scrape_papers(
        str(papers_dir),
        papers_path=str(papers_path),
        output_path=str(output_path),
        image_dir=str(image_root),
        mode='text-images',
        image_context='paper-text',
        image_extraction='pages',
        image_dpi=150,
        image_batch_size=1,
        vision_model='vision-model',
        vision_provider='local',
        vision_base_url='http://localhost:8000/v1',
        delete_images_after=True,
    )

    materials = pd.read_csv(output_path, index_col=0)
    papers = pd.read_csv(papers_path, index_col=0)
    image_dir = papers.loc[0, 'image_dir']
    image_paths = [path for batch in calls['image_batches'] for path in batch]
    assert FakeModelConfig.calls[0]['profile'] == 'text'
    assert FakeModelConfig.calls[1] == {
        'profile': 'vision',
        'name': 'vision-model',
        'provider': 'local',
        'base_url': 'http://localhost:8000/v1',
    }
    assert FakeModelConfig.required == ['vision']
    assert calls['text'] == 'single chunk'
    assert calls['contexts'] == ['text context', 'text context']
    assert calls['combine'][0] == [{'Name': 'text LLZO'}]
    assert calls['combine'][1] == [{'Name': 'image 1'}, {'Name': 'image 1'}]
    assert calls['image_extract']['strategy'] == 'pages'
    assert calls['image_extract']['dpi'] == 150
    assert materials['Name'].tolist() == ['combined LLZO']
    assert materials['Source'].tolist() == ['text+image']
    assert papers.loc[0, 'text_scrape_status'] == 'succeeded'
    assert papers.loc[0, 'image_scrape_status'] == 'succeeded'
    assert papers.loc[0, 'num_images'] == 2
    assert papers.loc[0, 'num_image_materials'] == 2
    assert image_dir == str(image_root / 'paper-1')
    assert all(not os.path.exists(path) for path in image_paths)


def test_scrape_papers_falls_back_to_separate_rows_when_combining_returns_no_records(tmp_path, monkeypatch):
    """
    Test fallback behavior when text-image reconciliation returns no records.

    This function performs the following steps:
    1. Writes matching text and PDF files for one paper.
    2. Replaces model and document helpers with local fakes that produce text and image records.
    3. Replaces reconciliation with a fake that returns an empty list.
    4. Calls `scrape_papers` in text-image mode.

    Asserts:
        - Text and image material rows are both preserved.
        - The empty reconciliation result is recorded in `last_error`.
        - Image rows include semicolon-separated image source paths.
    """
    papers_dir = tmp_path / 'papers'
    papers_dir.mkdir()
    (papers_dir / 'paper-1.txt').write_text('paper text')
    (papers_dir / 'paper-1.pdf').write_text('pdf bytes')
    papers_path = tmp_path / 'papers.csv'
    output_path = tmp_path / 'scraped.csv'
    image_path = tmp_path / 'image.png'
    image_path.write_text('image')
    write_papers_csv(papers_path, [{'paper_id': 'paper-1'}])

    monkeypatch.setattr(scrape, 'load_recipe', lambda recipe: sample_recipe())
    monkeypatch.setattr(scrape, 'ModelConfig', FakeModelConfig)
    monkeypatch.setattr(scrape, 'tqdm', FakeTqdm)
    monkeypatch.setattr(scrape, '_text_chunks', lambda text, model_config, prompt='': ['chunk'])
    monkeypatch.setattr(scrape, 'read_document_text', lambda path: 'text')
    monkeypatch.setattr(scrape, 'extract_pdf_images', lambda *_, **__: [str(image_path)])
    monkeypatch.setattr(scrape, 'scrape_text', lambda *_, **__: [{'Name': 'text material'}])
    monkeypatch.setattr(scrape, 'scrape_images', lambda *_, **__: [{'Name': 'image material'}])

    monkeypatch.setattr(scrape, 'combine_material_records', lambda *_, **__: [])

    scrape.scrape_papers(
        str(papers_dir),
        papers_path=str(papers_path),
        output_path=str(output_path),
        mode='text-images',
    )

    materials = pd.read_csv(output_path, index_col=0)
    papers = pd.read_csv(papers_path, index_col=0)
    assert materials['Name'].tolist() == ['text material', 'image material']
    assert materials['Source'].tolist() == ['text', 'image']
    assert materials.loc[1, 'Source path'] == str(image_path)
    assert 'Combining text and image results failed: reconciliation returned no material records' in papers.loc[
        0, 'last_error']


def test_scrape_papers_image_mode_writes_image_rows_reads_context_and_deletes_pdf(tmp_path, monkeypatch):
    """
    Test image-only scraping with paper-text context and paper cleanup.

    This function performs the following steps:
    1. Writes a papers CSV and matching PDF file.
    2. Replaces model, progress, PDF text, image extraction, and image analysis helpers with local fakes.
    3. Calls `scrape_papers` in image-only mode with paper cleanup enabled.
    4. Reloads the materials and papers CSV files.

    Asserts:
        - PDF text is read as context when text was not already loaded.
        - Image-derived material rows are written with image provenance.
        - Image scrape status and counts are updated.
        - The successfully scraped PDF source is deleted.
    """
    papers_dir = tmp_path / 'papers'
    papers_dir.mkdir()
    pdf_path = papers_dir / 'paper-1.pdf'
    pdf_path.write_text('pdf bytes')
    papers_path = tmp_path / 'papers.csv'
    output_path = tmp_path / 'scraped.csv'
    image_path = tmp_path / 'image.png'
    image_path.write_text('image')
    write_papers_csv(papers_path, [{'paper_id': 'paper-1', 'doi': '10.1/one'}])
    calls = {}

    monkeypatch.setattr(scrape, 'load_recipe', lambda recipe: sample_recipe())
    monkeypatch.setattr(scrape, 'ModelConfig', FakeModelConfig)
    monkeypatch.setattr(scrape, 'tqdm', FakeTqdm)
    monkeypatch.setattr(scrape, 'read_pdf_text', lambda path: f'context from {os.path.basename(path)}')
    monkeypatch.setattr(scrape, 'extract_pdf_images', lambda *_, **__: [str(image_path)])

    def fake_analyze_images(image_paths, recipe, model_config=None, context=None):
        calls['image_paths'] = image_paths
        calls['context'] = context
        return [{'Name': 'image-only material'}]

    monkeypatch.setattr(scrape, 'scrape_images', fake_analyze_images)

    scrape.scrape_papers(
        str(papers_dir),
        papers_path=str(papers_path),
        output_path=str(output_path),
        mode='images',
        image_context='paper-text',
        delete_papers_after=True,
    )

    materials = pd.read_csv(output_path, index_col=0)
    papers = pd.read_csv(papers_path, index_col=0)
    assert calls['image_paths'] == [str(image_path)]
    assert calls['context'] == 'context from paper-1.pdf'
    assert materials['Name'].tolist() == ['image-only material']
    assert materials['Source'].tolist() == ['image']
    assert materials['Source path'].tolist() == [str(image_path)]
    assert papers.loc[0, 'image_scrape_status'] == 'succeeded'
    assert papers.loc[0, 'num_images'] == 1
    assert papers.loc[0, 'num_image_materials'] == 1
    assert not pdf_path.exists()


def test_scrape_papers_skips_already_successful_image_stage(tmp_path, monkeypatch, capsys):
    """
    Test image scraping skip behavior for already successful rows.

    This function performs the following steps:
    1. Writes a papers CSV with an already successful image scrape status.
    2. Replaces model configuration and progress helpers with local fakes.
    3. Replaces image extraction with a fake that would fail if called.
    4. Calls `scrape_papers` in image-only mode.

    Asserts:
        - Image extraction is not called.
        - No materials file is written.
        - The skipped image-stage summary is printed.
    """
    papers_dir = tmp_path / 'papers'
    papers_dir.mkdir()
    papers_path = tmp_path / 'papers.csv'
    output_path = tmp_path / 'scraped.csv'
    write_papers_csv(papers_path, [{'paper_id': 'paper-1', 'image_scrape_status': 'succeeded'}])

    monkeypatch.setattr(scrape, 'load_recipe', lambda recipe: sample_recipe())
    monkeypatch.setattr(scrape, 'ModelConfig', FakeModelConfig)
    monkeypatch.setattr(scrape, 'tqdm', FakeTqdm)
    monkeypatch.setattr(
        scrape,
        'extract_pdf_images',
        lambda *_, **__: (_ for _ in ()).throw(AssertionError('image extraction should not run')),
    )

    scrape.scrape_papers(str(papers_dir), papers_path=str(papers_path), output_path=str(output_path), mode='images')

    output = capsys.readouterr().out
    assert not output_path.exists()
    assert 'Skipped already successful stages: text=0, images=1' in output


def test_scrape_papers_records_image_failure_when_pdf_is_missing(tmp_path, monkeypatch):
    """
    Test image scraping when no matching PDF exists.

    This function performs the following steps:
    1. Writes a papers CSV without creating a PDF file.
    2. Replaces recipe loading, model config, and progress bar with local fakes.
    3. Calls `scrape_papers` in image-only mode.
    4. Reloads the papers CSV.

    Asserts:
        - The image scrape status is marked as failed.
        - The missing-PDF error is recorded.
        - No materials CSV is created.
    """
    papers_dir = tmp_path / 'papers'
    papers_dir.mkdir()
    papers_path = tmp_path / 'papers.csv'
    output_path = tmp_path / 'scraped.csv'
    write_papers_csv(papers_path, [{'paper_id': 'paper-1'}])

    monkeypatch.setattr(scrape, 'load_recipe', lambda recipe: sample_recipe())
    monkeypatch.setattr(scrape, 'ModelConfig', FakeModelConfig)
    monkeypatch.setattr(scrape, 'tqdm', FakeTqdm)

    scrape.scrape_papers(str(papers_dir), papers_path=str(papers_path), output_path=str(output_path), mode='images')

    papers = pd.read_csv(papers_path, index_col=0)
    assert papers.loc[0, 'image_scrape_status'] == 'failed'
    assert 'No downloaded PDF file found for image analysis' in papers.loc[0, 'last_error']
    assert not output_path.exists()


def test_scrape_papers_records_image_failures(tmp_path, monkeypatch):
    """
    Test image scraping failure handling.

    This function performs the following steps:
    1. Writes a papers CSV and matching PDF file.
    2. Replaces image extraction with a fake returning no image paths.
    3. Calls `scrape_papers` in image-only mode.
    4. Reloads the papers CSV.

    Asserts:
        - Image scrape status is marked as failed.
        - The no-images error is recorded.
        - No materials CSV is created.
    """
    papers_dir = tmp_path / 'papers'
    papers_dir.mkdir()
    (papers_dir / 'paper-1.pdf').write_text('pdf bytes')
    papers_path = tmp_path / 'papers.csv'
    output_path = tmp_path / 'scraped.csv'
    write_papers_csv(papers_path, [{'paper_id': 'paper-1'}])

    monkeypatch.setattr(scrape, 'load_recipe', lambda recipe: sample_recipe())
    monkeypatch.setattr(scrape, 'ModelConfig', FakeModelConfig)
    monkeypatch.setattr(scrape, 'tqdm', FakeTqdm)
    monkeypatch.setattr(scrape, 'extract_pdf_images', lambda *_, **__: [])

    scrape.scrape_papers(str(papers_dir), papers_path=str(papers_path), output_path=str(output_path), mode='images')

    papers = pd.read_csv(papers_path, index_col=0)
    assert papers.loc[0, 'image_scrape_status'] == 'failed'
    assert 'No PDF images could be extracted or rendered' in papers.loc[0, 'last_error']
    assert not output_path.exists()
