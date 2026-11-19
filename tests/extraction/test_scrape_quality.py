"""Test structure-aware chunking, table text and checks inside the scrape workflow."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import paperminertoolkit.corpus.database as corpus
import paperminertoolkit.extraction.scrape as scrape
from tests.extraction.test_scrape import FakeModelConfig, FakeTqdm, sample_recipe, write_corpus

NL = chr(10)
JATS = """<article><body><sec><p>Body.</p><table-wrap id="t1"><label>Table 1</label><table>
<thead><tr><th>Name</th><th>Conductivity</th></tr></thead>
<tbody><tr><td>LLZO</td><td>0.5 mS</td></tr></tbody></table></table-wrap></sec></body></article>"""


def test_split_block_wraps_long_lines_and_repeats_table_headers() -> None:
    """Split oversized tables between rows with their header, and prose between words."""
    rows = [f'| M{index} | {index}.0 |' for index in range(20)]
    table = NL.join(['Table 1 Gaps', '| Material | Gap |', '| --- | --- |', *rows])

    pieces = scrape._split_block(table, 100)

    header = NL.join(['Table 1 Gaps', '| Material | Gap |', '| --- | --- |'])
    assert len(pieces) > 1
    assert all(piece.startswith(header + NL + "| M") and len(piece) <= 100 for piece in pieces)
    assert sum(piece.count('.0 |') for piece in pieces) == 20

    wide = NL.join(['| ' + 'x' * 60 + ' |', '| --- |', '| a |', '| b |'])
    assert scrape._split_block(wide, 70)[0].startswith('| ' + 'x' * 60)

    prose = ' '.join(['word'] * 50)
    pieces = scrape._split_block(prose, 40)
    assert all(len(piece) <= 40 for piece in pieces)
    assert ' '.join(pieces).split() == prose.split()


def test_overlap_repeats_a_short_prose_tail_but_never_a_table() -> None:
    """Carry the end of a paragraph into the next chunk, but not a table."""
    assert scrape._overlap('| a |' + NL + '| --- |', 1000) == ''
    assert scrape._overlap('Short.', 1000) == 'Short.'
    block = 'The first sentence is long enough. The gap is 1.5 eV here'
    tail = scrape._overlap(block, 400)
    assert block.endswith(tail) and 0 < len(tail) <= 20
    assert scrape._overlap('abcdefghijklmnopqrstuvwxyz', 400) == 'ghijklmnopqrstuvwxyz'


def test_fit_chunks_skips_blank_blocks_and_resplits_overflowing_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Split again when a packed chunk still exceeds the token budget."""
    monkeypatch.setattr(scrape, 'token_length', lambda text, model_config=None: len(text))
    text = NL * 2 + (NL * 2).join(['alpha ' * 10, 'beta ' * 10, 'gamma ' * 10]) + NL * 3

    chunks = scrape._fit_chunks(text, 40, 4.0, None)

    assert len(chunks) >= 3
    assert ' '.join(chunks).split().count('alpha') >= 10


def test_text_chunks_fit_dense_tables_by_remeasuring_overflowing_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every chunk within budget when numbers need far more tokens than prose."""
    monkeypatch.setattr(scrape, 'request_token_budget', lambda prompt, model_config: 100)

    def tokens(text: str, model_config: object = None) -> int:
        """Count prose at four characters a token and every digit as a token."""
        digits = sum(character.isdigit() for character in text)
        return digits + (len(text) - digits) // 4

    monkeypatch.setattr(scrape, 'token_length', tokens)
    prose = ''.join(f'Paragraph {chr(97 + index % 26)} of the methods section.{NL}' for index in range(30))
    table = ''.join(f'| {index:04d} | {index * 37 % 1000:03d} | {index * 91 % 1000:03d} |{NL}' for index in range(60))

    chunks = scrape._text_chunks(prose + NL + table, None)

    assert all(tokens(chunk) <= 100 for chunk in chunks)
    assert sum(line.startswith('| 0') for chunk in chunks for line in chunk.splitlines()) == 60


def test_write_dropped_appends_audit_rows_beside_the_output(tmp_path: Path) -> None:
    """Write nothing for no rows, and add a header only once."""
    output = tmp_path / 'scraped.csv'
    row = dict.fromkeys(scrape.DROPPED_COLUMNS, 'x')

    assert scrape._write_dropped([], output) == 0
    assert scrape._write_dropped([row], output) == 1
    assert scrape._write_dropped([row], output) == 1
    assert scrape.dropped_path(output) == str(tmp_path / 'scraped.dropped.csv')
    assert len(pd.read_csv(scrape.dropped_path(output))) == 2


def test_paper_tables_prefer_structured_documents_and_fall_back_to_the_pdf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use XML tables when stored, never mix in PDF tables, and fall back otherwise."""
    monkeypatch.setattr(scrape, 'pdf_tables_text',
                        lambda path, skip_regions=(): f'pdf tables from {path} skipping {list(skip_regions)}')
    monkeypatch.setattr(scrape, 'figure_regions', lambda path: ['fig-1'])
    db_path = tmp_path / 'papers.db'
    with corpus.connect(db_path) as conn:
        papers = [{'paper_id': name} for name in ('xml', 'empty-xml', 'html', 'none')]
        for paper in papers:
            corpus.upsert_paper(conn, paper)
        corpus.add_structured_document(conn, papers[0], JATS.encode(), document_format='jats', source='pubmed')
        corpus.add_structured_document(conn, papers[1], b'<article/>', document_format='jats', source='pubmed')
        corpus.add_structured_document(conn, papers[2], b'<html/>', document_format='html', source='web')

        assert '| LLZO | 0.5 mS |' in scrape._paper_tables(conn, 'xml', 'paper.pdf')
        assert scrape._paper_tables(conn, 'empty-xml', 'paper.pdf') == ''
        assert scrape._paper_tables(conn, 'html', 'paper.pdf') == 'pdf tables from paper.pdf skipping []'
        assert scrape._paper_tables(conn, 'html', 'paper.pdf', figure_text=False) == (
            "pdf tables from paper.pdf skipping ['fig-1']"
        )
        assert scrape._paper_tables(conn, 'none', None) == ''


def test_scrape_papers_appends_tables_merges_chunks_and_verifies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Send tables to the model, merge chunk records and audit what the checks drop."""
    papers_dir = tmp_path / 'papers'
    papers_dir.mkdir()
    (papers_dir / 'paper-1.txt').write_text('LLZO conducts at 1.2 mS.', encoding='utf-8')
    db_path = tmp_path / 'papers.db'
    output_path = tmp_path / 'scraped.csv'
    write_corpus(db_path, [{'paper_id': 'paper-1', 'doi': '10.1/one'}])
    with corpus.connect(db_path) as conn:
        corpus.add_structured_document(conn, {'paper_id': 'paper-1'}, JATS.encode(), document_format='jats',
                                       source='pubmed')
    recipe = {**sample_recipe(), 'checks': {'values in source': True, 'drop empty': True}}
    sent = []

    def fake_scrape_text(text: str, recipe: Mapping[str, Any], model_config: object = None) -> list[dict[str, str]]:
        """Return overlapping records from each chunk."""
        sent.append(text)
        if len(sent) == 1:
            return [{'Name': 'LLZO', 'Conductivity': '1.2 mS'}, {'Name': 'Ghost', 'Conductivity': '9.9 mS'}]
        return [{'Name': 'llzo', 'Conductivity': '0.5 mS'}]

    monkeypatch.setattr(scrape, 'load_recipe', lambda name: recipe)
    monkeypatch.setattr(scrape, 'ModelConfig', FakeModelConfig)
    monkeypatch.setattr(scrape, 'tqdm', FakeTqdm)
    monkeypatch.setattr(scrape, '_text_chunks', lambda text, model_config, prompt='': [text, 'second chunk'])
    monkeypatch.setattr(scrape, 'scrape_text', fake_scrape_text)

    scrape.scrape_papers(str(db_path), output_path=str(output_path), mode='text', verify=True)

    assert '=== TABLES ===' in sent[0] and '| LLZO | 0.5 mS |' in sent[0]
    materials = pd.read_csv(output_path, index_col=0)
    assert materials[['Name', 'Conductivity']].values.tolist() == [['LLZO', '1.2 mS']]
    audit = pd.read_csv(scrape.dropped_path(output_path))
    assert audit[['Record', 'Check']].values.tolist() == [['Ghost', 'values in source'], ['Ghost', 'drop empty']]
    assert 'Checks removed, edited or moved 2 items' in capsys.readouterr().out

    sent.clear()
    output_path.unlink()
    scrape.scrape_papers(str(db_path), output_path=str(output_path), mode='text', force=True, tables=False)
    assert '=== TABLES ===' not in sent[0]
    assert len(pd.read_csv(output_path, index_col=0)) == 2


def test_scrape_papers_asks_the_paper_question_and_moves_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ask the recipe's paper question with --verify and move fields on a no."""
    papers_dir = tmp_path / 'papers'
    papers_dir.mkdir()
    (papers_dir / 'paper-1.txt').write_text('LLZO conducts at 1.2 mS [3].', encoding='utf-8')
    (papers_dir / 'paper-2.txt').write_text('No materials here.', encoding='utf-8')
    db_path = tmp_path / 'papers.db'
    output_path = tmp_path / 'scraped.csv'
    write_corpus(db_path, [{'paper_id': 'paper-1', 'doi': '10.1/one'}, {'paper_id': 'paper-2', 'doi': '10.1/two'}])
    recipe = sample_recipe()
    recipe['search fields']['Cited'] = {'prompt': 'Cited conductivity.', 'example': '1e-3 S cm^-1'}
    recipe['checks'] = {'paper question': {'question': 'Own results?', 'if no': {'move': {'Conductivity': 'Cited'}}}}
    asked = []

    def fake_scrape_text(text: str, recipe: Mapping[str, Any], model_config: object = None) -> list[dict[str, str]]:
        """Return one record for the first paper only."""
        return [{'Name': 'LLZO', 'Conductivity': '1.2 mS', 'Cited': 'None'}] if 'LLZO' in text else []

    def fake_ask(text: str, question: str, model_config: object = None) -> bool:
        """Answer no, recording what was asked."""
        asked.append((text, question))
        return False

    monkeypatch.setattr(scrape, 'load_recipe', lambda name: recipe)
    monkeypatch.setattr(scrape, 'ModelConfig', FakeModelConfig)
    monkeypatch.setattr(scrape, 'tqdm', FakeTqdm)
    monkeypatch.setattr(scrape, 'scrape_text', fake_scrape_text)
    monkeypatch.setattr(scrape, 'ask_paper_question', fake_ask)

    scrape.scrape_papers(str(db_path), output_path=str(output_path), mode='text', verify=True)

    assert asked == [('LLZO conducts at 1.2 mS [3].', 'Own results?')]
    materials = pd.read_csv(output_path, index_col=0, keep_default_na=False)
    assert materials[['Name', 'Conductivity', 'Cited']].values.tolist() == [['LLZO', 'None', "['1.2 mS']"]]
    audit = pd.read_csv(scrape.dropped_path(output_path))
    assert audit[['Field', 'Check', 'Action', 'Detail']].values.tolist() == [
        ['Conductivity', 'paper question', 'moved', 'to Cited'],
    ]

    asked.clear()
    output_path.unlink()
    scrape.scrape_papers(str(db_path), output_path=str(output_path), mode='text', force=True)
    assert asked == []
