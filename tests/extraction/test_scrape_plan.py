"""Test scrape dry runs: selection, token estimates, and the no-side-effect guarantee."""

from __future__ import annotations

from contextlib import closing
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import pytest
import requests
from click.testing import CliRunner

import paperminertoolkit.corpus.database as corpus
import paperminertoolkit.extraction.scrape as scrape
import paperminertoolkit.extraction.scrape_plan as scrape_plan
import paperminertoolkit.extraction.tokenizer as tokenizer
from paperminertoolkit import cli
from paperminertoolkit.corpus import filtering
from paperminertoolkit.corpus.supplements import add_supplement


@dataclass
class _Config:
    """Model configuration with a provider whose tokens are character estimates."""

    provider: str = 'test'
    name: str = 'test-model'
    api_key: str | None = 'secret'
    base_url: str | None = None
    input_token_limit: int = 4000


def _recipe() -> dict[str, Any]:
    """Return a minimal recipe."""
    return {
        'record definition': {'subject': 'solid electrolytes', 'singular': 'material',
                              'plural': 'materials', 'unit': 'a composition',
                              'identity fields': ['Name']},
        'search fields': {'Name': {'prompt': 'Material name.', 'example': 'LLZO'}},
    }


def _forbid(name: str) -> Any:
    """Return a callable that fails the test if a side effect is attempted."""
    def forbidden(*args: Any, **kwargs: Any) -> NoReturn:
        """Fail on any call."""
        raise AssertionError(f'{name} was called during a dry run')
    return forbidden


@pytest.fixture
def planned_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a corpus covering every planning outcome and forbid side effects."""
    db_path = tmp_path / 'papers.db'
    with closing(corpus.connect(db_path)) as conn, conn:
        rows = [
            {'paper_id': 'long', 'title': 'Keep long'},
            {'paper_id': 'short', 'title': 'Keep short'},
            {'paper_id': 'done', 'title': 'Keep done', 'text_scrape_status': 'succeeded'},
            {'paper_id': 'nothing', 'title': 'Keep nothing'},
            {'paper_id': 'excluded', 'title': 'Drop me'},
        ]
        for row in rows:
            corpus.upsert_paper(conn, row)
        long_text = tmp_path / 'long.txt'
        long_text.write_text('LLZO conductivity table 1.2e-3 ' * 4000, encoding='utf-8')
        short_text = tmp_path / 'short.txt'
        short_text.write_text('LiPON thin film.', encoding='utf-8')
        corpus.add_asset(conn, rows[0], long_text, role='text', kind='text', mime_type='text/plain')
        corpus.add_asset(conn, rows[1], short_text, role='text', kind='text', mime_type='text/plain')
        add_supplement(conn, rows[1], b'LATP, 4e-4 S/cm', original_filename='si.txt',
                       metadata={'label': 'Table S1'})
    rules = tmp_path / 'rules.json'
    rules.write_text(json.dumps({'name': 'keep', 'fields': ['title'],
                                 'include': [{'name': 'keep', 'pattern': 'Keep'}]}),
                     encoding='utf-8')
    filtering.apply_regex_filter(db_path, rules)
    monkeypatch.setattr(scrape_plan, 'load_recipe', lambda recipe: _recipe())
    monkeypatch.setattr(scrape_plan.ModelConfig, 'from_profile',
                        classmethod(lambda cls, *args, **kwargs: _Config()))
    for module, name in [(scrape, 'scrape_text'), (scrape, 'scrape_images'),
                         (scrape, 'combine_material_records')]:
        monkeypatch.setattr(module, name, _forbid(name))
    monkeypatch.setattr(requests.Session, 'request', _forbid('requests.Session.request'))
    monkeypatch.setattr(requests, 'post', _forbid('requests.post'))
    return db_path


def test_plan_mirrors_selection_and_measures_text_without_side_effects(
    planned_corpus: Path,
) -> None:
    """Report skips, failures, chunked inputs, and SI while leaving the corpus untouched."""
    before = planned_corpus.read_bytes()

    plan = scrape_plan.plan_scrape(str(planned_corpus), 'sse', mode='text')

    assert planned_corpus.read_bytes() == before
    assert plan['dry_run'] is True
    assert plan['papers_selected'] == 4
    assert plan['papers_filtered_out'] == 1
    papers = {paper['paper_id']: paper for paper in plan['papers']}
    assert set(papers) == {'long', 'short', 'done', 'nothing'}
    assert papers['done']['stages']['text'] == {'action': 'skip', 'reason': 'already succeeded'}
    assert papers['nothing']['stages']['text']['action'] == 'fail'
    long_stage = papers['long']['stages']['text']
    assert long_stage['action'] == 'scrape'
    assert long_stage['source'] == 'text'
    assert long_stage['requests'] > 1
    assert long_stage['largest_request_tokens'] <= _Config.input_token_limit
    assert papers['short']['stages']['text']['requests'] == 1
    member = papers['short']['supplements'][0]['stages']['text']['members'][0]
    assert member['action'] == 'scrape' and member['requests'] == 1
    totals = plan['totals']
    assert totals == {
        'papers_to_scrape': 2, 'skipped_done': 1, 'skipped_missing_source': 1,
        'requests': long_stage['requests'] + 1 + 1,
        'input_tokens': long_stage['input_tokens'] + papers['short']['stages']['text']['input_tokens']
        + member['input_tokens'],
        'image_inputs': 0, 'supplement_files_to_scrape': 1, 'supplement_files_skipped': 0,
    }
    assert [item['paper_id'] for item in plan['overflows']] == ['long']
    assert plan['tokenizer'] == 'estimate'


def test_plan_requests_match_the_real_chunker(planned_corpus: Path) -> None:
    """Count the same number of requests a real run would send for the long paper."""
    plan = scrape_plan.plan_scrape(str(planned_corpus), 'sse', paper_ids=['long'])
    text = (Path(planned_corpus).parent / 'long.txt').read_text(encoding='utf-8')
    prompt = scrape.build_scrape_prompt(_recipe(), source='text')
    assert plan['papers'][0]['stages']['text']['requests'] == len(
        scrape._text_chunks(text, _Config(), prompt=prompt))


def test_plan_honours_ids_force_count_and_ignore_filters(planned_corpus: Path) -> None:
    """Apply the same selection options as a real scrape."""
    named = scrape_plan.plan_scrape(str(planned_corpus), 'sse', paper_ids=['excluded', 'done'],
                                    force=True)
    assert [paper['paper_id'] for paper in named['papers']] == ['done', 'excluded']
    assert named['papers'][0]['stages']['text']['action'] == 'fail'
    assert named['papers_filtered_out'] == 0
    everything = scrape_plan.plan_scrape(str(planned_corpus), 'sse', ignore_filters=True,
                                         scrape_count=2, scrape_order='paper-id')
    assert [paper['paper_id'] for paper in everything['papers']] == ['done', 'excluded']


def test_plan_counts_anthropic_tokens_offline(
    planned_corpus: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the character estimate rather than Anthropic's token-counting endpoint."""
    monkeypatch.setattr(scrape_plan.ModelConfig, 'from_profile',
                        classmethod(lambda cls, *args, **kwargs: _Config(provider='anthropic')))
    plan = scrape_plan.plan_scrape(str(planned_corpus), 'sse', paper_ids=['short'])
    assert plan['tokenizer'] == 'estimate'
    assert plan['papers'][0]['stages']['text']['requests'] == 1


def test_offline_counting_loads_local_tokenizers_from_disk_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never let a dry run download a Hugging Face tokenizer."""
    calls = []

    class _Tokenizer:
        """Tokenizer double that splits on spaces."""

        def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
            """Return one token per word."""
            return text.split()

    def load(model: str, local_files_only: bool = False) -> _Tokenizer:
        """Record how the tokenizer was loaded."""
        calls.append(local_files_only)
        return _Tokenizer()

    monkeypatch.setattr(tokenizer, '_auto_tokenizer', load)
    config = _Config(provider='local')
    assert tokenizer.count_text_tokens('a b c', model_config=config) == 3
    with tokenizer.offline_token_counting():
        assert tokenizer.count_text_tokens('a b c', model_config=config) == 3
    assert calls == [False, True]
    assert tokenizer.token_count_method(config, offline=True) == 'transformers'
    assert tokenizer.token_count_method(_Config(provider='anthropic')) == 'anthropic'
    assert tokenizer.token_count_method(_Config(provider='anthropic'), offline=True) == 'estimate'


def test_plan_refuses_an_unmigrated_corpus(tmp_path: Path) -> None:
    """Ask for a migration instead of writing to an old corpus."""
    db_path = tmp_path / 'old.db'
    import sqlite3
    with sqlite3.connect(db_path) as conn:
        conn.execute('PRAGMA user_version = 1')
    with pytest.raises(RuntimeError, match='does not match the supported version'):
        scrape_plan.plan_scrape(str(db_path))
    with pytest.raises(FileNotFoundError):
        scrape_plan.plan_scrape(str(tmp_path / 'missing.db'))


def test_cli_dry_run_never_starts_a_scrape(
    planned_corpus: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Route --dry-run to the planner, in JSON and as a readable summary."""
    monkeypatch.setattr(cli, 'scrape_papers', _forbid('scrape_papers'))
    before = planned_corpus.read_bytes()
    runner = CliRunner()
    result = runner.invoke(cli.main, ['scrape', str(planned_corpus), '--dry-run', '--json'])
    assert result.exit_code == 0, result.output
    plan = json.loads(result.stdout)
    assert plan['dry_run'] is True and plan['papers_selected'] == 4

    human = runner.invoke(cli.main, ['scrape', str(planned_corpus), '--dry-run'])
    assert human.exit_code == 0, human.output
    assert 'Dry run: text scrape' in human.stdout
    assert 'would scrape 2, skip 1 done, and fail 1 with no downloaded source' in human.stdout
    assert 'long text:' in human.stdout
    assert planned_corpus.read_bytes() == before
