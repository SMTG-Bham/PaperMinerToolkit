"""Test scrape helpers and corpus extraction workflows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import os
from pathlib import Path
from typing import Any, Self

import pandas as pd
import pytest

import paperminer.corpus as corpus
import paperminer.scrape as scrape
from paperminer.compression import CompressionConfig


def sample_recipe() -> dict[str, Any]:
    """Return a minimal recipe for scrape unit tests."""
    return {
        'record definition': {
            'subject': 'solid electrolytes',
            'singular': 'material',
            'plural': 'materials',
            'unit': 'a distinct solid-electrolyte composition or sample',
            'identity fields': ['Name'],
        },
        'search fields': {
            'Name': {'prompt': 'Material name.', 'example': 'LLZO'},
            'Conductivity': {'prompt': 'Conductivity.', 'example': '1e-3 S cm^-1'},
        },
    }


def write_corpus(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write paper rows and matching local files to a corpus database for scrape unit tests."""
    papers_dir = path.parent / 'papers'
    with corpus.connect(path) as conn:
        for row in rows:
            corpus.upsert_paper(conn, row)
            paper_id = row.get('paper_id')
            if row.get('abstract'):
                corpus.add_asset(conn, row, row['abstract'], role='abstract', kind='text', mime_type='text/plain')
            text_path = papers_dir / f'{paper_id}.txt'
            if text_path.is_file():
                corpus.add_asset(conn, row, text_path, role='text', kind='text', mime_type='text/plain')
            pdf_path = papers_dir / f'{paper_id}.pdf'
            if pdf_path.is_file():
                corpus.add_asset(conn, row, pdf_path, role='pdf', kind='pdf', mime_type='application/pdf')


def read_corpus(path: Path) -> pd.DataFrame:
    """Read paper rows from a corpus database as a DataFrame for scrape unit tests."""
    with corpus.connect(path) as conn:
        return pd.DataFrame(corpus.paper_rows(conn))


class FakeTqdm:
    """Minimal progress-bar replacement for scrape unit tests."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize progress update tracking."""
        self.updates = 0

    def __enter__(self) -> Self:
        """Return the fake progress bar from its context manager."""
        return self

    def __exit__(self, *_: object) -> bool:
        """Leave the progress context without suppressing errors."""
        return False

    def update(self, amount: int) -> None:
        """Add an amount to the recorded progress."""
        self.updates += amount


class FakeModelConfig:
    """Minimal model configuration replacement for scrape unit tests."""

    calls = []
    required = []

    def __init__(
        self,
        profile: str,
        name: str | None = None,
        provider: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Store fake model profile and connection values."""
        self.profile = profile
        self.name = name or f'{profile}-model'
        self.provider = provider
        self.base_url = base_url

    @classmethod
    def from_profile(
        cls,
        profile: str,
        name: str | None = None,
        provider: str | None = None,
        base_url: str | None = None,
    ) -> Self:
        """Record profile arguments and return a fake configuration."""
        cls.calls.append({
            'profile': profile,
            'name': name,
            'provider': provider,
            'base_url': base_url,
        })
        return cls(profile, name=name, provider=provider, base_url=base_url)

    def require(self, capability: str) -> None:
        """Record a required model capability."""
        self.required.append(capability)


def test_text_chunks_uses_model_token_estimate_to_split_long_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test model-aware text chunking for short and long inputs."""
    text_config = FakeModelConfig('text', name='model')
    reserve_calls = []

    def fake_reserve(
        prompt: str,
        model_config: FakeModelConfig | None = None,
        buffer_tokens: int = 500,
    ) -> int:
        """Record token reservation arguments and return a fixed reserve."""
        reserve_calls.append({
            'prompt': prompt,
            'model_config': model_config,
            'buffer_tokens': buffer_tokens,
        })
        return 750

    def fake_limit(model_config: FakeModelConfig | None = None, reserve_tokens: int = 2000) -> int:
        """Validate the token reserve and return a fixed context limit."""
        assert reserve_tokens == 750
        return 120000

    def short_count(text: str, model_config: FakeModelConfig | None = None) -> int:
        """Return a token count that fits the configured context limit."""
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


def test_material_file_and_image_helpers_handle_common_inputs(tmp_path: Path) -> None:
    """Test material, file, path, and image batching helpers."""
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


def test_select_papers_orders_and_limits_corpus_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test paper ordering, limiting, and option validation."""
    papers = [
        {'paper_id': 'paper-2', 'title': 'Beta', 'publication_date': '2024-01-01'},
        {'paper_id': 'paper-1', 'title': 'Alpha', 'publication_date': '2026-01-01'},
        {'paper_id': 'paper-3', 'title': 'Gamma', 'publication_date': '2025-01-01'},
    ]

    assert [row['paper_id'] for row in scrape._select_papers(papers)] == ['paper-2', 'paper-1', 'paper-3']
    assert [row['paper_id'] for row in scrape._select_papers(papers, 'publication-asc')] == [
        'paper-2',
        'paper-3',
        'paper-1',
    ]
    assert [row['paper_id'] for row in scrape._select_papers(papers, 'publication-desc', 2)] == [
        'paper-1',
        'paper-3',
    ]
    assert [row['paper_id'] for row in scrape._select_papers(papers, 'title')] == ['paper-1', 'paper-2', 'paper-3']
    assert [row['paper_id'] for row in scrape._select_papers(papers, 'paper-id')] == ['paper-1', 'paper-2', 'paper-3']

    monkeypatch.setattr(scrape.random, 'shuffle', lambda rows: rows.reverse())
    assert [row['paper_id'] for row in scrape._select_papers(papers, 'random')] == ['paper-3', 'paper-1', 'paper-2']

    with pytest.raises(ValueError, match='scrape_order must be one of'):
        scrape._select_papers(papers, scrape_order='bad')
    with pytest.raises(ValueError, match='positive integer'):
        scrape._select_papers(papers, scrape_count=0)


def test_scrape_papers_rejects_invalid_modes(tmp_path: Path) -> None:
    """Test validation of scrape, image, compression, and selection options."""
    papers_dir = tmp_path / 'papers'
    papers_dir.mkdir()
    db_path = tmp_path / 'papers.db'
    write_corpus(db_path, [{'paper_id': 'paper-1'}])

    with pytest.raises(ValueError, match='mode must be one of'):
        scrape.scrape_papers(str(db_path), mode='bad')
    with pytest.raises(ValueError, match='image_context must be one of'):
        scrape.scrape_papers(str(db_path), image_context='bad')
    with pytest.raises(ValueError, match='image_extraction must be one of'):
        scrape.scrape_papers(str(db_path), image_extraction='bad')
    with pytest.raises(ValueError, match='compression_scope must be one of'):
        scrape.scrape_papers(str(db_path), compression_scope='bad')
    with pytest.raises(ValueError, match='compression_mode must be one of'):
        scrape.scrape_papers(str(db_path), compression_mode='bad')
    with pytest.raises(ValueError, match='compression_ratio must be'):
        scrape.scrape_papers(str(db_path), compression_ratio='1.5')
    with pytest.raises(ValueError, match='scrape_order must be one of'):
        scrape.scrape_papers(str(db_path), scrape_order='bad')
    with pytest.raises(ValueError, match='scrape_count must be a positive integer'):
        scrape.scrape_papers(str(db_path), scrape_count=0)


def test_scrape_papers_text_mode_writes_materials_updates_status_and_preserves_import_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test text scraping output, status updates, and source preservation."""
    papers_dir = tmp_path / 'papers'
    papers_dir.mkdir()
    text_path = papers_dir / 'paper-1.txt'
    text_path.write_text('paper text')
    db_path = tmp_path / 'papers.db'
    output_path = tmp_path / 'scraped.csv'
    write_corpus(db_path, [
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
    monkeypatch.setattr(scrape, 'maybe_compress_text', lambda text, prompt, model_config, config: text)

    def fake_analyze_text(
        text: str,
        recipe: Mapping[str, Any],
        model_config: FakeModelConfig | None = None,
    ) -> list[dict[str, str]]:
        """Record analyzed chunks and return a material record."""
        calls.setdefault('chunks', []).append(text)
        calls.setdefault('configs', []).append(model_config)
        return [{'Name': f'material from {text}'}]

    monkeypatch.setattr(scrape, 'scrape_text', fake_analyze_text)

    scrape.scrape_papers(
        str(db_path),
        output_path=str(output_path),
        mode='text',
        recipe='sse',
        model='text-model',
        provider='local',
        base_url='http://localhost:8000/v1',
    )

    materials = pd.read_csv(output_path, index_col=0)
    papers = read_corpus(db_path)
    captured = capsys.readouterr()
    output = captured.out
    assert calls['chunks'] == ['chunk one', 'chunk two']
    assert all(config.name == 'text-model' for config in calls['configs'])
    assert materials['Name'].tolist() == ['material from chunk one', 'material from chunk two']
    assert materials['Source'].tolist() == ['text', 'text']
    assert materials['Paper id'].tolist() == ['paper-1', 'paper-1']
    assert papers.loc[0, 'text_scrape_status'] == 'succeeded'
    assert papers.loc[0, 'num_text_materials'] == 2
    assert papers.loc[0, 'num_text_chunks'] == 2
    assert papers.loc[1, 'text_scrape_status'] == 'succeeded'
    assert text_path.exists()
    assert FakeModelConfig.calls == [{
        'profile': 'text',
        'name': 'text-model',
        'provider': 'local',
        'base_url': 'http://localhost:8000/v1',
    }]
    assert 'Skipped already successful stages: abstracts=0, text=1, images=0' in output
    assert 'text for paper paper-1 was split into 2 independent model requests' in captured.err
    assert 'Results from separate chunks are not automatically reconciled' in captured.err
    assert 'Chunking warning: 1 paper input was split into 2 independent model requests' in captured.err


def test_force_rescrape_replaces_previous_chunk_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Record the latest chunk plan rather than retaining an earlier split count."""
    papers_dir = tmp_path / 'papers'
    papers_dir.mkdir()
    (papers_dir / 'paper-1.txt').write_text('paper text')
    db_path = tmp_path / 'papers.db'
    write_corpus(db_path, [{
        'paper_id': 'paper-1',
        'text_scrape_status': 'succeeded',
        'num_text_chunks': 4,
    }])

    monkeypatch.setattr(scrape, 'load_recipe', lambda recipe: sample_recipe())
    monkeypatch.setattr(scrape, 'ModelConfig', FakeModelConfig)
    monkeypatch.setattr(scrape, 'tqdm', FakeTqdm)
    monkeypatch.setattr(scrape, 'read_document_text', lambda path: 'short paper text')
    monkeypatch.setattr(scrape, 'maybe_compress_text', lambda text, prompt, model_config, config: text)
    monkeypatch.setattr(scrape, '_text_chunks', lambda text, model_config, prompt='': [text])
    monkeypatch.setattr(scrape, 'scrape_text', lambda text, recipe, model_config=None: [])

    scrape.scrape_papers(str(db_path), mode='text', force=True, output_path=str(tmp_path / 'scraped.csv'))

    papers = read_corpus(db_path)
    captured = capsys.readouterr()
    assert papers.loc[0, 'text_scrape_status'] == 'succeeded'
    assert papers.loc[0, 'num_text_chunks'] == 1
    assert 'was split into' not in captured.err


def test_scrape_papers_abstract_mode_writes_materials_and_updates_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test abstract scraping output and independent status updates."""
    db_path = tmp_path / 'papers.db'
    output_path = tmp_path / 'scraped.csv'
    write_corpus(db_path, [{
        'paper_id': 'paper-abstract',
        'doi': '10.1/abstract',
        'publication_date': '2026',
        'abstract': 'abstract text',
    }])
    calls = {}
    FakeModelConfig.calls = []
    monkeypatch.setattr(scrape, 'load_recipe', lambda recipe: sample_recipe())
    monkeypatch.setattr(scrape, 'ModelConfig', FakeModelConfig)
    monkeypatch.setattr(scrape, 'tqdm', FakeTqdm)
    monkeypatch.setattr(scrape, '_text_chunks', lambda text, model_config, prompt='': [text])
    monkeypatch.setattr(scrape, 'read_document_text', lambda path: 'abstract text from corpus')
    monkeypatch.setattr(scrape, 'maybe_compress_text', lambda text, prompt, model_config, config: text)

    def fake_scrape_text(
        text: str,
        recipe: Mapping[str, Any],
        model_config: FakeModelConfig | None = None,
    ) -> list[dict[str, str]]:
        """Record abstract analysis inputs and return a material record."""
        calls['text'] = text
        calls['model_config'] = model_config
        return [{'Name': 'abstract material'}]

    monkeypatch.setattr(scrape, 'scrape_text', fake_scrape_text)

    scrape.scrape_papers(str(db_path), output_path=str(output_path), mode='abstract')

    materials = pd.read_csv(output_path, index_col=0)
    papers = read_corpus(db_path)
    assert calls['text'] == 'abstract text from corpus'
    assert calls['model_config'].profile == 'text'
    assert materials['Name'].tolist() == ['abstract material']
    assert materials['Source'].tolist() == ['abstract']
    assert materials['Source path'].tolist() == ['corpus:abstract']
    assert papers.loc[0, 'abstract_scrape_status'] == 'succeeded'
    assert papers.loc[0, 'text_scrape_status'] == 'pending'
    assert papers.loc[0, 'num_abstract_materials'] == 1
    assert papers.loc[0, 'num_abstract_chunks'] == 1


def test_scrape_papers_applies_count_and_order_before_scraping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test ordering and count limits in the main scrape loop."""
    papers_dir = tmp_path / 'papers'
    papers_dir.mkdir()
    for paper_id in ['paper-3', 'paper-1', 'paper-2']:
        (papers_dir / f'{paper_id}.txt').write_text(f'text for {paper_id}')
    db_path = tmp_path / 'papers.db'
    output_path = tmp_path / 'scraped.csv'
    write_corpus(db_path, [
        {'paper_id': 'paper-3'},
        {'paper_id': 'paper-1'},
        {'paper_id': 'paper-2'},
    ])

    monkeypatch.setattr(scrape, 'load_recipe', lambda recipe: sample_recipe())
    monkeypatch.setattr(scrape, 'ModelConfig', FakeModelConfig)
    monkeypatch.setattr(scrape, 'tqdm', FakeTqdm)
    monkeypatch.setattr(scrape, '_text_chunks', lambda text, model_config, prompt='': [text])
    monkeypatch.setattr(scrape, 'read_document_text', lambda path: os.path.basename(os.path.dirname(path)))
    monkeypatch.setattr(scrape, 'maybe_compress_text', lambda text, prompt, model_config, config: text)
    monkeypatch.setattr(scrape, 'scrape_text', lambda text, recipe, model_config=None: [{'Name': text}])

    scrape.scrape_papers(
        str(db_path),
        output_path=str(output_path),
        mode='text',
        scrape_count=2,
        scrape_order='paper-id',
    )

    materials = pd.read_csv(output_path, index_col=0)
    papers = read_corpus(db_path).sort_values('paper_id').reset_index(drop=True)
    assert materials['Paper id'].tolist() == ['paper-1', 'paper-2']
    assert papers.loc[0, 'text_scrape_status'] == 'succeeded'
    assert papers.loc[1, 'text_scrape_status'] == 'succeeded'
    assert papers.loc[2, 'text_scrape_status'] == 'pending'


def test_scrape_papers_records_text_failures_when_source_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test text scrape failure handling when no source asset exists."""
    papers_dir = tmp_path / 'papers'
    papers_dir.mkdir()
    db_path = tmp_path / 'papers.db'
    output_path = tmp_path / 'scraped.csv'
    write_corpus(db_path, [{'paper_id': 'missing-paper'}])
    monkeypatch.setattr(scrape, 'load_recipe', lambda recipe: sample_recipe())
    monkeypatch.setattr(scrape, 'ModelConfig', FakeModelConfig)
    monkeypatch.setattr(scrape, 'tqdm', FakeTqdm)

    scrape.scrape_papers(str(db_path), output_path=str(output_path))

    papers = read_corpus(db_path)
    output = capsys.readouterr().out
    assert papers.loc[0, 'text_scrape_status'] == 'failed'
    assert 'No downloaded text or PDF asset found' in papers.loc[0, 'last_error']
    assert not output_path.exists()
    assert 'No new scraped material rows were written' in output


def test_scrape_papers_records_abstract_failures_when_source_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test abstract scrape failure handling when no source asset exists."""
    db_path = tmp_path / 'papers.db'
    output_path = tmp_path / 'scraped.csv'
    write_corpus(db_path, [{'paper_id': 'missing-abstract'}])
    monkeypatch.setattr(scrape, 'load_recipe', lambda recipe: sample_recipe())
    monkeypatch.setattr(scrape, 'ModelConfig', FakeModelConfig)
    monkeypatch.setattr(scrape, 'tqdm', FakeTqdm)

    scrape.scrape_papers(str(db_path), output_path=str(output_path), mode='abstract')

    papers = read_corpus(db_path)
    assert papers.loc[0, 'abstract_scrape_status'] == 'failed'
    assert 'No downloaded abstract asset found' in papers.loc[0, 'last_error']
    assert not output_path.exists()


def test_scrape_papers_compresses_text_before_chunking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that scrape-time text compression precedes chunking."""
    papers_dir = tmp_path / 'papers'
    papers_dir.mkdir()
    (papers_dir / 'paper-1.txt').write_text('paper text')
    db_path = tmp_path / 'papers.db'
    output_path = tmp_path / 'scraped.csv'
    write_corpus(db_path, [{'paper_id': 'paper-1'}])
    calls = {}

    monkeypatch.setattr(scrape, 'load_recipe', lambda recipe: sample_recipe())
    monkeypatch.setattr(scrape, 'ModelConfig', FakeModelConfig)
    monkeypatch.setattr(scrape, 'tqdm', FakeTqdm)
    monkeypatch.setattr(scrape, 'read_document_text', lambda path: 'full paper text')

    def fake_compress(
        text: str,
        prompt: str,
        model_config: FakeModelConfig,
        config: CompressionConfig,
    ) -> str:
        """Record compression inputs and return compressed text."""
        calls['compress'] = {
            'text': text,
            'prompt_contains': 'paper text' in prompt,
            'scope': config.scope,
            'mode': config.mode,
            'ratio': config.ratio,
            'content_detection': config.content_detection,
        }
        return 'compressed paper text'

    def fake_chunks(text: str, model_config: FakeModelConfig, prompt: str = '') -> list[str]:
        """Record chunking input and return one compressed chunk."""
        calls['chunk_text'] = text
        return ['compressed chunk']

    def fake_scrape_text(
        text: str,
        recipe: Mapping[str, Any],
        model_config: FakeModelConfig | None = None,
    ) -> list[dict[str, str]]:
        """Record analyzed text and return a material record."""
        calls['analyzed_text'] = text
        return [{'Name': 'compressed material'}]

    monkeypatch.setattr(scrape, 'maybe_compress_text', fake_compress)
    monkeypatch.setattr(scrape, '_text_chunks', fake_chunks)
    monkeypatch.setattr(scrape, 'scrape_text', fake_scrape_text)

    scrape.scrape_papers(
        str(db_path),
        output_path=str(output_path),
        compression_scope='text',
        compression_mode='always',
        compression_ratio='0.4',
        compression_content_detection=False,
    )

    assert calls['compress'] == {
        'text': 'full paper text',
        'prompt_contains': True,
        'scope': 'text',
        'mode': 'always',
        'ratio': 0.4,
        'content_detection': False,
    }
    assert calls['chunk_text'] == 'compressed paper text'
    assert calls['analyzed_text'] == 'compressed chunk'


def test_scrape_papers_text_images_combines_results_and_cleans_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test combined scraping, reconciliation, and image cleanup."""
    papers_dir = tmp_path / 'papers'
    papers_dir.mkdir()
    text_path = papers_dir / 'paper-1.txt'
    pdf_path = papers_dir / 'paper-1.pdf'
    text_path.write_text('paper text')
    pdf_path.write_text('pdf bytes')
    db_path = tmp_path / 'papers.db'
    output_path = tmp_path / 'scraped.csv'
    image_root = tmp_path / 'images'
    write_corpus(db_path, [{'paper_id': 'paper-1', 'doi': '10.1/one', 'publication_date': '2024'}])
    FakeModelConfig.calls = []
    FakeModelConfig.required = []
    calls = {}

    monkeypatch.setattr(scrape, 'load_recipe', lambda recipe: sample_recipe())
    monkeypatch.setattr(scrape, 'ModelConfig', FakeModelConfig)
    monkeypatch.setattr(scrape, 'tqdm', FakeTqdm)
    monkeypatch.setattr(scrape, '_text_chunks', lambda text, model_config, prompt='': ['single chunk'])
    monkeypatch.setattr(scrape, 'read_document_text', lambda path: 'text context')

    def fake_extract_pdf_images(
        pdf_path_arg: str,
        output_dir: str,
        prefix: str,
        strategy: str,
        dpi: int,
    ) -> list[str]:
        """Create fake extracted images and record extraction options."""
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

    def fake_analyze_text(
        text: str,
        recipe: Mapping[str, Any],
        model_config: FakeModelConfig | None = None,
    ) -> list[dict[str, str]]:
        """Record analyzed text and return a text-derived record."""
        calls['text'] = text
        return [{'Name': 'text LLZO'}]

    def fake_analyze_images(
        image_paths: Sequence[str],
        recipe: Mapping[str, Any],
        model_config: FakeModelConfig | None = None,
        context: str | None = None,
        compression_config: CompressionConfig | None = None,
    ) -> list[dict[str, str]]:
        """Record an image batch and return an image-derived record."""
        calls.setdefault('image_batches', []).append(image_paths)
        calls.setdefault('contexts', []).append(context)
        calls.setdefault('image_compression', []).append(compression_config)
        return [{'Name': f'image {len(image_paths)}'}]

    def fake_combine(
        text_materials: Sequence[Mapping[str, Any]],
        image_materials: Sequence[Mapping[str, Any]],
        recipe: Mapping[str, Any],
        model_config: FakeModelConfig | None = None,
    ) -> list[dict[str, str]]:
        """Record source records and return a reconciled record."""
        calls['combine'] = (text_materials, image_materials, model_config.name)
        return [{'Name': 'combined LLZO'}]

    monkeypatch.setattr(scrape, 'extract_pdf_images', fake_extract_pdf_images)
    monkeypatch.setattr(scrape, 'scrape_text', fake_analyze_text)
    monkeypatch.setattr(scrape, 'scrape_images', fake_analyze_images)
    monkeypatch.setattr(scrape, 'combine_material_records', fake_combine)

    scrape.scrape_papers(
        str(db_path),
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
    papers = read_corpus(db_path)
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
    assert [config.scope for config in calls['image_compression']] == ['none', 'none']
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


def test_scrape_papers_falls_back_to_separate_rows_when_combining_returns_no_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test fallback rows when text-image reconciliation returns nothing."""
    papers_dir = tmp_path / 'papers'
    papers_dir.mkdir()
    (papers_dir / 'paper-1.txt').write_text('paper text')
    (papers_dir / 'paper-1.pdf').write_text('pdf bytes')
    db_path = tmp_path / 'papers.db'
    output_path = tmp_path / 'scraped.csv'
    image_path = tmp_path / 'image.png'
    image_path.write_text('image')
    write_corpus(db_path, [{'paper_id': 'paper-1'}])

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
        str(db_path),
        output_path=str(output_path),
        mode='text-images',
    )

    materials = pd.read_csv(output_path, index_col=0)
    papers = read_corpus(db_path)
    assert materials['Name'].tolist() == ['text material', 'image material']
    assert materials['Source'].tolist() == ['text', 'image']
    assert materials.loc[1, 'Source path'] == str(image_path)
    assert 'Combining text and image results failed: reconciliation returned no material records' in papers.loc[
        0, 'last_error']


def test_scrape_papers_image_mode_writes_image_rows_reads_context_and_preserves_import_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test image scraping with text context and source preservation."""
    papers_dir = tmp_path / 'papers'
    papers_dir.mkdir()
    pdf_path = papers_dir / 'paper-1.pdf'
    pdf_path.write_text('pdf bytes')
    db_path = tmp_path / 'papers.db'
    output_path = tmp_path / 'scraped.csv'
    image_path = tmp_path / 'image.png'
    image_path.write_text('image')
    write_corpus(db_path, [{'paper_id': 'paper-1', 'doi': '10.1/one'}])
    calls = {}

    monkeypatch.setattr(scrape, 'load_recipe', lambda recipe: sample_recipe())
    monkeypatch.setattr(scrape, 'ModelConfig', FakeModelConfig)
    monkeypatch.setattr(scrape, 'tqdm', FakeTqdm)
    monkeypatch.setattr(scrape, 'read_pdf_text', lambda path: f'context from {os.path.basename(path)}')
    monkeypatch.setattr(scrape, 'extract_pdf_images', lambda *_, **__: [str(image_path)])

    def fake_analyze_images(
        image_paths: Sequence[str],
        recipe: Mapping[str, Any],
        model_config: FakeModelConfig | None = None,
        context: str | None = None,
        compression_config: CompressionConfig | None = None,
    ) -> list[dict[str, str]]:
        """Record image analysis inputs and return a material record."""
        calls['image_paths'] = image_paths
        calls['context'] = context
        calls['compression_config'] = compression_config
        return [{'Name': 'image-only material'}]

    monkeypatch.setattr(scrape, 'scrape_images', fake_analyze_images)

    scrape.scrape_papers(
        str(db_path),
        output_path=str(output_path),
        mode='images',
        image_context='paper-text',
    )

    materials = pd.read_csv(output_path, index_col=0)
    papers = read_corpus(db_path)
    assert calls['image_paths'] == [str(image_path)]
    assert calls['context'] == 'context from paper-1.pdf'
    assert calls['compression_config'].scope == 'none'
    assert materials['Name'].tolist() == ['image-only material']
    assert materials['Source'].tolist() == ['image']
    assert materials['Source path'].tolist() == [str(image_path)]
    assert papers.loc[0, 'image_scrape_status'] == 'succeeded'
    assert papers.loc[0, 'num_images'] == 1
    assert papers.loc[0, 'num_image_materials'] == 1
    assert pdf_path.exists()


def test_scrape_papers_skips_already_successful_image_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test skipping an already successful image stage."""
    papers_dir = tmp_path / 'papers'
    papers_dir.mkdir()
    db_path = tmp_path / 'papers.db'
    output_path = tmp_path / 'scraped.csv'
    write_corpus(db_path, [{'paper_id': 'paper-1', 'image_scrape_status': 'succeeded'}])

    monkeypatch.setattr(scrape, 'load_recipe', lambda recipe: sample_recipe())
    monkeypatch.setattr(scrape, 'ModelConfig', FakeModelConfig)
    monkeypatch.setattr(scrape, 'tqdm', FakeTqdm)
    monkeypatch.setattr(
        scrape,
        'extract_pdf_images',
        lambda *_, **__: (_ for _ in ()).throw(AssertionError('image extraction should not run')),
    )

    scrape.scrape_papers(str(db_path), output_path=str(output_path), mode='images')

    output = capsys.readouterr().out
    assert not output_path.exists()
    assert 'Skipped already successful stages: abstracts=0, text=0, images=1' in output


def test_scrape_papers_records_image_failure_when_pdf_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test image scrape failure handling when no PDF exists."""
    papers_dir = tmp_path / 'papers'
    papers_dir.mkdir()
    db_path = tmp_path / 'papers.db'
    output_path = tmp_path / 'scraped.csv'
    write_corpus(db_path, [{'paper_id': 'paper-1'}])

    monkeypatch.setattr(scrape, 'load_recipe', lambda recipe: sample_recipe())
    monkeypatch.setattr(scrape, 'ModelConfig', FakeModelConfig)
    monkeypatch.setattr(scrape, 'tqdm', FakeTqdm)

    scrape.scrape_papers(str(db_path), output_path=str(output_path), mode='images')

    papers = read_corpus(db_path)
    assert papers.loc[0, 'image_scrape_status'] == 'failed'
    assert 'No downloaded PDF asset found for image analysis' in papers.loc[0, 'last_error']
    assert not output_path.exists()


def test_scrape_papers_records_image_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test image scrape failure handling when extraction yields no images."""
    papers_dir = tmp_path / 'papers'
    papers_dir.mkdir()
    (papers_dir / 'paper-1.pdf').write_text('pdf bytes')
    db_path = tmp_path / 'papers.db'
    output_path = tmp_path / 'scraped.csv'
    write_corpus(db_path, [{'paper_id': 'paper-1'}])

    monkeypatch.setattr(scrape, 'load_recipe', lambda recipe: sample_recipe())
    monkeypatch.setattr(scrape, 'ModelConfig', FakeModelConfig)
    monkeypatch.setattr(scrape, 'tqdm', FakeTqdm)
    monkeypatch.setattr(scrape, 'extract_pdf_images', lambda *_, **__: [])

    scrape.scrape_papers(str(db_path), output_path=str(output_path), mode='images')

    papers = read_corpus(db_path)
    assert papers.loc[0, 'image_scrape_status'] == 'failed'
    assert 'No PDF images could be extracted or rendered' in papers.loc[0, 'last_error']
    assert not output_path.exists()
