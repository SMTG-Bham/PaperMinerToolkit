"""Test merging and checking an existing scrape output."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pymupdf
import pytest

import paperminertoolkit.corpus.database as corpus
from paperminertoolkit.extraction import postprocess

RECIPE = {
    'record definition': {
        'subject': 'materials', 'singular': 'material', 'plural': 'materials', 'unit': 'a material',
        'identity fields': ['Material'],
    },
    'search fields': {
        'Material': {'prompt': 'Material.', 'example': '<material>'},
        'Gaps': {'prompt': 'Gaps.', 'example': [{'value': '<number> eV'}]},
    },
    'checks': {'values in source': True, 'drop empty': True},
}

JATS = """<article><body><sec><p>Body text.</p>
<table-wrap id="t1"><label>Table 1</label><table>
<thead><tr><th>Material</th><th>Gap</th></tr></thead>
<tbody><tr><td>ZrTaN3</td><td>0.45 eV</td></tr></tbody></table></table-wrap></sec></body></article>"""


def gap(value: str) -> dict[str, str]:
    """Build one band-gap item."""
    return {'value': value, 'method_or_source': 'None'}


def row(paper_id: str, material: str, gaps: Any, source: str = 'text') -> dict[str, Any]:
    """Build one scrape-output row."""
    return {'Material': material, 'Gaps': gaps, 'Paper id': paper_id, 'doi': f'10.1/{paper_id}', 'Source': source}


def build_corpus(tmp_path: Path) -> Path:
    """Write a corpus with text+XML, PDF-only, abstract-only and asset-less papers."""
    db_path = tmp_path / 'papers.db'
    pdf_path = tmp_path / 'paper-2.pdf'
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), 'LaWN3 has a band gap of 2.04 eV.')
    document.save(pdf_path)
    document.close()
    text_path = tmp_path / 'paper-1.txt'
    text_path.write_text('ZrTaN3 has an indirect gap. Abstract-free body.', encoding='utf-8')
    with corpus.connect(db_path) as conn:
        papers = [{'paper_id': f'paper-{index}', 'doi': f'10.1/paper-{index}'} for index in range(1, 4)]
        for paper in papers:
            corpus.upsert_paper(conn, paper)
        corpus.add_asset(conn, papers[0], text_path, role='text', kind='text', mime_type='text/plain')
        corpus.add_asset(conn, papers[0], 'The gap is 1.1 eV.', role='abstract', kind='text', mime_type='text/plain')
        corpus.add_structured_document(conn, papers[0], JATS.encode(), document_format='jats', source='pubmed')
        corpus.add_asset(conn, papers[1], pdf_path, role='pdf', kind='pdf', mime_type='application/pdf')
    return db_path


def test_postprocess_merges_checks_and_reports_unchecked_papers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Merge chunk duplicates, drop values not in the paper, and keep image rows."""
    monkeypatch.setattr(postprocess, 'load_recipe', lambda recipe: RECIPE)
    db_path = build_corpus(tmp_path)
    in_path = tmp_path / 'scraped.csv'
    rows = [
        row('paper-1', 'ZrTaN3', [gap('0.45 eV')]),
        row('paper-1', 'zrtan3', [gap('0.45 eV'), gap('0.92 eV')]),
        row('paper-1', 'ZrTaN3', [gap('1.1 eV'), gap('5.5 eV')], source='abstract'),
        row('paper-1', 'ZrTaN3', [gap('7.7 eV')], source='image'),
        row('paper-2', 'LaWN3', [gap('2.04 eV'), gap('1.26 eV')]),
        row('paper-2', 'Ghost', [gap('1.35 eV')]),
        row('paper-3', 'NoText', [gap('3.0 eV')]),
        row('paper-9', 'Unknown', [gap('3.0 eV')]),
    ]
    frame = pd.DataFrame(rows)
    frame.loc[len(frame)] = {'Material': 'Odd', 'Gaps': '[unparsed', 'Paper id': 'paper-9', 'doi': 'x', 'Source': 'image'}
    frame.loc[len(frame)] = {'Material': 'Json', 'Gaps': '["2.0 eV"]', 'Paper id': 'paper-9', 'doi': 'x',
                             'Source': 'image'}
    frame.to_csv(in_path)
    stale_audit = tmp_path / 'scraped.postprocessed.dropped.csv'
    stale_audit.write_text('stale')

    summary = postprocess.postprocess_scrape(db_path, 'band_gap', in_path)

    assert summary.output_path == str(tmp_path / 'scraped.postprocessed.csv')
    assert summary.dropped_path == str(stale_audit)
    assert (summary.records_in, summary.records_out) == (10, 8)
    assert summary.unchecked_papers == ('paper-3', 'paper-9')
    output = pd.read_csv(summary.output_path, index_col=0)
    assert output['Material'].tolist() == ['ZrTaN3', 'ZrTaN3', 'ZrTaN3', 'LaWN3', 'NoText', 'Unknown', 'Odd', 'Json']
    assert output['Gaps'].tolist()[:4] == [
        str([gap('0.45 eV')]), str([gap('1.1 eV')]), str([gap('7.7 eV')]), str([gap('2.04 eV')]),
    ]
    assert output['Gaps'].tolist()[6:] == ['[unparsed', "['2.0 eV']"]
    audit = pd.read_csv(summary.dropped_path)
    assert audit['Check'].tolist() == ['values in source'] * 4 + ['drop empty']
    assert summary.audit_rows == 5


def test_postprocess_can_merge_without_checks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the checks and the corpus text when verification is off."""
    monkeypatch.setattr(postprocess, 'load_recipe', lambda recipe: RECIPE)
    db_path = build_corpus(tmp_path)
    in_path = tmp_path / 'scraped.csv'
    pd.DataFrame([row('paper-1', 'A', [gap('9 eV')]), row('paper-1', 'a', [gap('9.0 eV')])]).to_csv(in_path)
    out_path = tmp_path / 'out.csv'

    summary = postprocess.postprocess_scrape(db_path, 'band_gap', in_path, out_path, verify=False)
    assert (summary.records_out, summary.audit_rows, summary.unchecked_papers) == (1, 0, ())

    summary = postprocess.postprocess_scrape(db_path, 'band_gap', in_path, out_path, merge=False, verify=False)
    assert summary.records_out == 2
