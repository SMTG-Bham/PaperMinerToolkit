"""Test scrape dry runs: selection, token estimates, and the no-side-effect guarantee."""

from __future__ import annotations

from contextlib import closing
import io
import json
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import pytest
import requests
from click.testing import CliRunner
from PIL import Image

import paperminertoolkit.corpus.database as corpus
import paperminertoolkit.extraction.scrape as scrape
import paperminertoolkit.extraction.scrape_plan as scrape_plan
import paperminertoolkit.extraction.tokenizer as tokenizer
from paperminertoolkit import cli
from paperminertoolkit.corpus import filtering
from paperminertoolkit.corpus.supplements import add_supplement, set_supplement_extraction_status


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


def _archive(files: Mapping[str, bytes]) -> bytes:
    """Return an in-memory ZIP archive holding the given members."""
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return stream.getvalue()


def _png() -> bytes:
    """Return a tiny PNG figure."""
    stream = io.BytesIO()
    Image.new('RGB', (3, 2)).save(stream, format='PNG')
    return stream.getvalue()


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


@pytest.mark.parametrize(('option', 'message'), [
    ({'mode': 'video'}, '^mode must be one of: .*text-images'),
    ({'image_extraction': 'ocr'}, '^image_extraction must be one of: .*layout'),
    ({'scrape_order': 'citations'}, '^scrape_order must be one of: .*paper-id'),
])
def test_plan_rejects_invalid_options_before_opening_the_corpus(
    tmp_path: Path, option: dict[str, str], message: str,
) -> None:
    """Name the valid choices for a bad option instead of reporting a missing corpus."""
    with pytest.raises(ValueError, match=message):
        scrape_plan.plan_scrape(str(tmp_path / 'missing.db'), **option)


def test_plan_predicts_every_supplement_outcome_a_real_run_would_reach(planned_corpus: Path) -> None:
    """Skip, fail, or scrape each SI file and archive member as a real text scrape would."""
    paper = {'paper_id': 'si', 'title': 'Keep SI'}
    bundle_members = {
        'good.txt': b'LLZO, 1e-3 S/cm', 'nested.zip': _archive({'inner.txt': b'inner'}),
        'figure.png': _png(), 'binary.txt': b'abc\x00def', 'blank.txt': b' \n\t ',
        'new.txt': b'LATP, 4e-4 S/cm',
    }
    with closing(corpus.connect(planned_corpus)) as conn, conn:
        mirrored = add_supplement(conn, paper, b'LiPON, 2e-6 S/cm', original_filename='mirrored.txt')
        corpus.link_asset(conn, 'si', mirrored, role='supplement', source='mirror',
                          original_filename='mirror.txt')
        add_supplement(conn, paper, b'corrupt zip', original_filename='bad.zip')
        done = add_supplement(conn, paper, b'LLTO, 1e-3 S/cm', original_filename='done.txt')
        set_supplement_extraction_status(conn, 'si', done, 'text', 'succeeded')
        bundle = add_supplement(conn, paper, _archive(bundle_members), original_filename='bundle.zip')
        set_supplement_extraction_status(conn, 'si', bundle, 'text', 'partial',
                                         members={'good.txt': 'succeeded'})
    before = planned_corpus.read_bytes()

    plan = scrape_plan.plan_scrape(str(planned_corpus), 'sse', mode='text', paper_ids=['si'])

    assert planned_corpus.read_bytes() == before
    (si,) = plan['papers']
    assert si['stages']['text']['action'] == 'fail'
    assert [item['blob_id'] for item in si['supplements']].count(mirrored) == 1
    stages = {item['label']: item['stages']['text'] for item in si['supplements']}
    assert set(stages) == {'mirrored.txt', 'bad.zip', 'done.txt', 'bundle.zip'}
    assert stages['bad.zip']['action'] == 'fail'
    assert 'not a zip file' in stages['bad.zip']['reason']
    assert stages['done.txt'] == {'action': 'skip', 'reason': 'already succeeded'}
    (mirrored_member,) = stages['mirrored.txt']['members']
    assert mirrored_member['action'] == 'scrape' and mirrored_member['requests'] == 1
    members = {entry['member']: entry for entry in stages['bundle.zip']['members']}
    assert list(members) == list(bundle_members)
    assert {key: (entry['action'], entry.get('reason')) for key, entry in members.items()} == {
        'good.txt': ('skip', 'already succeeded'),
        'nested.zip': ('skip', 'unsupported format'),
        'figure.png': ('skip', 'no text content'),
        'binary.txt': ('fail', 'Supplement contains binary data rather than supported text'),
        'blank.txt': ('fail', 'no extractable text'),
        'new.txt': ('scrape', None),
    }
    assert members['new.txt']['requests'] == 1
    assert plan['totals'] == {
        'papers_to_scrape': 1, 'skipped_done': 0, 'skipped_missing_source': 1,
        'requests': 2, 'input_tokens': mirrored_member['input_tokens'] + members['new.txt']['input_tokens'],
        'image_inputs': 0, 'supplement_files_to_scrape': 2, 'supplement_files_skipped': 1,
    }
    assert plan['overflows'] == []


def test_plan_flags_supplements_needing_several_or_too_many_requests(
    planned_corpus: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scrape SI split into a few requests, but refuse SI over the request limit."""
    with closing(corpus.connect(planned_corpus)) as conn, conn:
        add_supplement(conn, {'paper_id': 'si', 'title': 'Keep SI'}, b'LLZO, 1e-3 S/cm\n' * 50,
                       original_filename='table.csv', metadata={'label': 'Table S2'})
    monkeypatch.setattr(scrape_plan, '_text_chunks',
                        lambda text, model_config, prompt='': ['LLZO, 1e-3 S/cm\n' * 10] * 3)

    split = scrape_plan.plan_scrape(str(planned_corpus), 'sse', paper_ids=['si'])

    (member,) = split['papers'][0]['supplements'][0]['stages']['text']['members']
    assert member['action'] == 'scrape' and member['requests'] == 3
    overflow = {'paper_id': 'si', 'item': 'supplement Table S2: table.csv', 'requests': 3,
                'text_tokens': member['text_tokens']}
    assert split['overflows'] == [{**overflow, 'scraped': True}]
    assert split['totals']['requests'] == 3
    assert split['totals']['input_tokens'] == member['input_tokens']
    assert split['totals']['supplement_files_to_scrape'] == 1
    assert split['totals']['papers_to_scrape'] == 1

    monkeypatch.setattr(scrape_plan, 'MAX_SUPPLEMENT_TEXT_REQUESTS', 2)
    refused = scrape_plan.plan_scrape(str(planned_corpus), 'sse', paper_ids=['si'])

    (member,) = refused['papers'][0]['supplements'][0]['stages']['text']['members']
    assert member['action'] == 'skip'
    assert member['reason'] == 'too large: 3 requests exceed the limit of 2'
    assert refused['overflows'] == [{**overflow, 'scraped': False}]
    assert {key: refused['totals'][key] for key in (
        'requests', 'input_tokens', 'supplement_files_to_scrape', 'papers_to_scrape')} == {
        'requests': 0, 'input_tokens': 0, 'supplement_files_to_scrape': 0, 'papers_to_scrape': 0}


def test_plan_image_mode_checks_pdfs_and_lists_si_images_without_measuring_them(
    planned_corpus: Path, tmp_path: Path,
) -> None:
    """Skip drawn papers, fail papers without a PDF, and count every image input."""
    import fitz

    pdf = tmp_path / 'paper.pdf'
    with fitz.open() as document:
        document.new_page(width=100, height=100)
        document.save(str(pdf))
    drawn = {'paper_id': 'drawn', 'title': 'Keep drawn', 'image_scrape_status': 'succeeded'}
    no_pdf = {'paper_id': 'no-pdf', 'title': 'Keep no PDF'}
    with_pdf = {'paper_id': 'with-pdf', 'title': 'Keep PDF'}
    with closing(corpus.connect(planned_corpus)) as conn, conn:
        for row in (drawn, no_pdf, with_pdf):
            corpus.upsert_paper(conn, row)
        corpus.add_asset(conn, with_pdf, pdf, role='pdf', kind='pdf', mime_type='application/pdf')
        add_supplement(conn, drawn, _png(), original_filename='figure.png', metadata={'label': 'Figure S1'})
        add_supplement(conn, drawn, b'LLZO synthesis details', original_filename='methods.txt')

    plan = scrape_plan.plan_scrape(str(planned_corpus), 'sse', mode='images',
                                   paper_ids=['drawn', 'no-pdf', 'with-pdf'])

    papers = {paper['paper_id']: paper for paper in plan['papers']}
    assert papers['drawn']['stages'] == {'image': {'action': 'skip', 'reason': 'already succeeded'}}
    assert papers['no-pdf']['stages'] == {'image': {'action': 'fail', 'reason': 'no downloaded PDF'}}
    assert papers['with-pdf']['stages'] == {
        'image': {'action': 'scrape', 'reason': 'image tokens are not estimated'}}
    assert {item['label']: item['stages'] for item in papers['drawn']['supplements']} == {
        'Figure S1': {'image': {'action': 'scrape', 'members': [
            {'member': 'figure.png', 'action': 'scrape', 'reason': 'image tokens are not estimated'}]}},
        'methods.txt': {'image': {'action': 'scrape', 'members': [
            {'member': 'methods.txt', 'action': 'skip', 'reason': 'no image content'}]}},
    }
    assert {key: plan['totals'][key] for key in ('papers_to_scrape', 'image_inputs', 'requests',
                                                 'input_tokens')} == {
        'papers_to_scrape': 2, 'image_inputs': 2, 'requests': 0, 'input_tokens': 0}
