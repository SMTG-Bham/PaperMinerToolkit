"""Exercise the ``pmt figures`` commands on real corpora and PDFs."""

from __future__ import annotations

import json
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, NoReturn

import pytest
from click.testing import CliRunner

import paperminertoolkit._figures_cli as figures_cli
from paperminertoolkit import cli
from paperminertoolkit.corpus import database, filtering
from paperminertoolkit.workflows import figures
from tests.workflows.test_figures import PNG, ImageResponse, ImageSession, _add_jats, _write_figure_pdf


def _run(*arguments: str) -> dict[str, Any]:
    """Run a figures command with --json and return its result."""
    result = CliRunner().invoke(cli.main, ['figures', *arguments, '--json'])
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Build papers with a figure PDF, without a PDF, and with a structured figure."""
    pdf = tmp_path / 'figure.pdf'
    _write_figure_pdf(pdf)
    path = tmp_path / 'papers.db'
    rows = {'with-pdf': {'paper_id': 'with-pdf', 'title': 'Keep PDF'},
            'no-pdf': {'paper_id': 'no-pdf', 'title': 'Keep bare'},
            'structured': {'paper_id': 'structured', 'title': 'Keep XML'}}
    with closing(database.connect(path)) as conn:
        for row in rows.values():
            database.upsert_paper(conn, row)
        database.add_asset(conn, rows['with-pdf'], pdf, role='pdf', kind='pdf',
                           mime_type='application/pdf')
        database.add_asset(conn, rows['structured'], pdf, role='pdf', kind='pdf',
                           mime_type='application/pdf')
        database.add_figure_asset(conn, rows['structured'], PNG, figure_id='fig-1',
                                  caption='Publisher figure.', source='pubmed',
                                  source_url='https://repo.example/fig-1.png', mime_type='image/png',
                                  metadata={'figure_label': 'Figure 1'})
    return path


def _figure_count(path: Path, paper_id: str) -> int:
    """Count the figure links stored for one paper."""
    with closing(database.connect(path)) as conn:
        return len(database.get_figure_assets(conn, paper_id))


def test_detect_keeps_one_source_per_paper_and_forces_without_duplicates(db_path: Path) -> None:
    """Detect where needed, skip structured or done papers, and replace on --force."""
    first = _run('detect', str(db_path))
    assert first['statuses'] == {'detected': 1, 'none_found': 0, 'skipped_structured': 1,
                                 'skipped_existing': 0, 'no_pdf': 1, 'failed': 0}
    assert first['figures_stored'] == 1
    again = _run('detect', str(db_path))
    assert again['statuses']['skipped_existing'] == 1 and again['figures_stored'] == 0
    forced = _run('detect', str(db_path), '--paper-id', 'with-pdf', '--force', '--dpi', '100')
    assert forced['statuses']['detected'] == 1 and forced['figures_replaced'] == 1
    assert _figure_count(db_path, 'with-pdf') == 1
    assert _figure_count(db_path, 'structured') == 1


def test_detect_respects_filters_unless_papers_are_named(db_path: Path, tmp_path: Path) -> None:
    """Follow active corpus filters as scrape does, with --paper-id bypassing them."""
    rules = tmp_path / 'xml.json'
    rules.write_text(json.dumps({'name': 'xml', 'fields': ['title'],
                                 'include': [{'name': 'xml', 'pattern': 'XML'}]}), encoding='utf-8')
    filtering.apply_regex_filter(db_path, rules)
    filtered = _run('detect', str(db_path))
    assert filtered['papers'] == 1 and filtered['papers_filtered_out'] == 2
    named = _run('detect', str(db_path), '--paper-id', 'with-pdf')
    assert named['statuses']['detected'] == 1


def test_list_and_export_read_stored_figures(db_path: Path, tmp_path: Path) -> None:
    """List figures read-only with paging, and export them without overwriting."""
    _run('detect', str(db_path))
    before = db_path.read_bytes()
    listed = _run('list', str(db_path))
    assert listed['total'] == 2
    by_paper = {figure['paper_id']: figure for figure in listed['figures']}
    assert by_paper['with-pdf']['source'] == 'pdf-layout'
    assert by_paper['with-pdf']['region_detected'] is True
    assert by_paper['with-pdf']['label'] == 'Figure 1'
    assert by_paper['structured']['source'] == 'pubmed'
    assert by_paper['structured']['region_detected'] is None
    page = _run('list', str(db_path), '--limit', '1', '--offset', '1')
    assert len(page['figures']) == 1 and page['total'] == 2
    only = _run('list', str(db_path), '--source', 'pubmed')
    assert [figure['paper_id'] for figure in only['figures']] == ['structured']
    assert db_path.read_bytes() == before

    outdir = tmp_path / 'figures'
    exported = _run('export', str(db_path), str(outdir))
    assert exported['count'] == 2
    paths = {Path(item['path']).relative_to(outdir).as_posix() for item in exported['files']}
    assert paths == {'with-pdf/pdf-figure-1.png', 'structured/fig-1.png'}
    assert (outdir / 'structured' / 'fig-1.png').read_bytes() == PNG
    refused = CliRunner().invoke(cli.main, ['figures', 'export', str(db_path), str(outdir), '--json'])
    assert refused.exit_code == 1
    assert 'already exist' in json.loads(refused.stdout)['error']['message']
    assert _run('export', str(db_path), str(outdir), '--force', '--paper-id', 'structured')['count'] == 1


@pytest.mark.parametrize(('strategy', 'expected'), [('layout', 1), ('pages', 1), ('embedded', 0)])
def test_render_runs_each_image_strategy_on_a_pdf(
    strategy: str, expected: int, tmp_path: Path,
) -> None:
    """Write detected regions, page renders, or embedded rasters from a PDF file."""
    pdf = tmp_path / 'figure.pdf'
    _write_figure_pdf(pdf)
    rendered = _run('render', str(pdf), str(tmp_path / 'out'), '--strategy', strategy)
    assert rendered['count'] == expected
    assert all(Path(image['path']).is_file() for image in rendered['images'])
    if strategy == 'layout':
        assert rendered['images'][0]['label'] == 'Figure 1'
        assert rendered['images'][0]['region_detected'] is True
        assert rendered['tables_detected'] == 0
    else:
        assert rendered['tables_detected'] is None


def test_structured_download_replaces_pdf_detected_figures(tmp_path: Path) -> None:
    """Keep the publisher's figure and drop the PDF crop of the same paper."""
    db_path = tmp_path / 'corpus.db'
    _add_jats(db_path, b'''<article xmlns:xlink="http://www.w3.org/1999/xlink"><body>
        <fig id="fig-1"><label>Figure 1</label><caption><p>Caption.</p></caption>
          <graphic id="g1" xlink:href="https://cdn.example/one.png"/></fig></body></article>''')
    pdf = tmp_path / 'figure.pdf'
    _write_figure_pdf(pdf)
    with closing(database.connect(db_path)) as conn:
        paper = database.paper_rows(conn)[0]
        database.add_asset(conn, paper, pdf, role='pdf', kind='pdf', mime_type='application/pdf')
        assert figures.detect_paper_figures(conn, paper)['status'] == 'detected'
    result = figures.download_paper_figures(db_path, 'paper:figures',
                                            session=ImageSession([ImageResponse()]))
    assert result['downloaded'] == 1 and result['pdf_layout_replaced'] == 1
    with closing(database.connect(db_path)) as conn:
        assert database.figure_sources(conn, 'paper:figures') == {'pubmed'}


def test_download_command_selects_papers_with_structured_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Visit only papers whose structured documents can reference figures."""
    db_path = tmp_path / 'corpus.db'
    _add_jats(db_path, b'<article><body/></article>')
    with closing(database.connect(db_path)) as conn:
        database.upsert_paper(conn, {'paper_id': 'plain', 'title': 'No XML'})
    visited = []

    def download(path: str, paper_id: str, *, force: bool = False) -> dict[str, Any]:
        """Record the visit and report one stored figure."""
        visited.append(paper_id)
        return {'paper_id': paper_id, 'downloaded': 1, 'skipped': 0, 'failed': 0, 'failures': [],
                'pdf_layout_replaced': 0}

    monkeypatch.setattr(figures_cli, 'download_paper_figures', download)
    summary = _run('download', str(db_path))
    assert visited == ['paper:figures']
    assert summary['papers'] == 1 and summary['downloaded'] == 1
    named = _run('download', str(db_path), '--paper-id', 'plain')
    assert named['papers'] == 0 and named['papers_without_structured_documents'] == 1


def _text(*arguments: str) -> Any:
    """Run a figures command without --json and return the click result."""
    return CliRunner().invoke(cli.main, ['figures', *arguments])


def test_download_prints_a_summary_with_failures_and_reports_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Summarise the run on stdout, list failures on stderr, and stop cleanly on an error."""
    db_path = tmp_path / 'corpus.db'
    _add_jats(db_path, b'<article><body/></article>')

    def download(path: str, paper_id: str, *, force: bool = False) -> dict[str, Any]:
        """Report one stored, two existing, and one refused figure."""
        return {'paper_id': paper_id, 'downloaded': 1, 'skipped': 2, 'failed': 1,
                'failures': ['https://cdn.example/two.png: HTTP 404'], 'pdf_layout_replaced': 1}

    monkeypatch.setattr(figures_cli, 'download_paper_figures', download)
    result = _text('download', str(db_path), '--progress', 'none')
    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == ('Figures from structured documents of 1 papers: 1 downloaded, '
                                     '2 already stored, 1 failed; 1 PDF-detected figures replaced.')
    assert '  paper:figures: https://cdn.example/two.png: HTTP 404' in result.stderr.splitlines()

    def exhausted(path: str, paper_id: str, *, force: bool = False) -> NoReturn:
        """Fail as a provider error that escapes the per-figure handling does."""
        raise RuntimeError('daily credit budget is exhausted')

    monkeypatch.setattr(figures_cli, 'download_paper_figures', exhausted)
    failed = _text('download', str(db_path), '--progress', 'none')
    assert failed.exit_code == 1
    assert 'daily credit budget is exhausted' in failed.stderr
    assert 'Figures from structured documents' not in failed.stdout


def test_detect_prints_statuses_and_failures_and_reports_errors(
    db_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Count every detection outcome in words, list failures, and stop cleanly on an error."""
    with closing(database.connect(db_path)) as conn:
        database.upsert_paper(conn, {'paper_id': 'broken', 'title': 'Keep broken'})
    detect = figures_cli.detect_paper_figures

    def detect_or_fail(conn: sqlite3.Connection, paper: dict[str, Any], **options: Any) -> dict[str, Any]:
        """Fail the paper named broken and detect the others for real."""
        if paper['paper_id'] == 'broken':
            return {'paper_id': 'broken', 'status': 'failed', 'stored': 0, 'replaced': 0,
                    'failures': ['cannot open the stored PDF']}
        return detect(conn, paper, **options)

    monkeypatch.setattr(figures_cli, 'detect_paper_figures', detect_or_fail)
    result = _text('detect', str(db_path), '--progress', 'none')
    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == [
        'Detected figures in 1 of 4 papers: 1 figures stored, 0 replaced.',
        'Skipped: 1 with structured figures, 0 already detected, 1 without a PDF; '
        '0 had no figure captions; 1 failed.',
    ]
    assert '  broken: cannot open the stored PDF' in result.stderr.splitlines()

    def locked(conn: sqlite3.Connection, paper: dict[str, Any], **options: Any) -> NoReturn:
        """Fail as a corpus another process holds does."""
        raise sqlite3.OperationalError('database is locked')

    monkeypatch.setattr(figures_cli, 'detect_paper_figures', locked)
    failed = _text('detect', str(db_path), '--progress', 'none')
    assert failed.exit_code == 1
    assert 'database is locked' in failed.stderr


def test_list_and_export_print_text_and_refuse_an_older_corpus(db_path: Path, tmp_path: Path) -> None:
    """Describe each stored figure in a line, export in words, and refuse an unmigrated corpus."""
    _run('detect', str(db_path))
    with closing(database.connect(db_path)) as conn:
        assert database.set_figure_extraction_status(conn, 'structured', 'fig-1', 'succeeded')
    listed = _text('list', str(db_path))
    assert listed.exit_code == 0, listed.output
    lines = listed.stdout.splitlines()
    assert lines[0] == '2 figures; showing 2 from offset 0.'
    assert set(lines[1:]) == {
        'with-pdf pdf-figure-1 [pdf-layout, region; not scraped] Figure 1: '
        'Conductivity map for the sample.',
        'structured fig-1 [pubmed; succeeded] Figure 1: Publisher figure.',
    }
    outdir = tmp_path / 'figures'
    exported = _text('export', str(db_path), str(outdir))
    assert exported.exit_code == 0, exported.output
    assert exported.stdout.strip() == f'Exported 2 figures to {outdir}.'
    assert (outdir / 'structured' / 'fig-1.png').read_bytes() == PNG

    older = tmp_path / 'older.db'
    shutil.copyfile(db_path, older)
    with closing(sqlite3.connect(older)) as conn:
        conn.execute(f'PRAGMA user_version = {database.SCHEMA_VERSION - 1}')
    for arguments in (['list', str(older)], ['export', str(older), str(tmp_path / 'older-figures')]):
        refused = _text(*arguments)
        assert refused.exit_code == 1
        assert 'does not match the supported version' in refused.stderr
    assert not (tmp_path / 'older-figures').exists()


def test_render_prints_each_image_and_reports_an_unreadable_pdf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Name each written image, with its label for layout, and stop cleanly on a bad PDF."""
    pdf = tmp_path / 'figure.pdf'
    _write_figure_pdf(pdf)
    outdir = str(tmp_path / 'layout')
    layout = _text('render', str(pdf), outdir)
    assert layout.exit_code == 0, layout.output
    head, image = layout.stdout.splitlines()
    assert head == f'Wrote 1 images to {outdir} (layout).'
    assert image.startswith('  ') and image.endswith(' Figure 1')
    assert Path(image.strip().removesuffix(' Figure 1')).is_file()
    pages = _text('render', str(pdf), str(tmp_path / 'pages'), '--strategy', 'pages')
    assert pages.exit_code == 0, pages.output
    head, image = pages.stdout.splitlines()
    assert head == f'Wrote 1 images to {tmp_path / "pages"} (pages).'
    assert Path(image.strip()).is_file()

    def unreadable(*_: object, **__: object) -> NoReturn:
        """Fail as a PDF without a readable page tree does."""
        raise ValueError('cannot read the PDF page tree')

    monkeypatch.setattr(figures_cli, 'detect_pdf_layout', unreadable)
    failed = _text('render', str(pdf), str(tmp_path / 'failed'))
    assert failed.exit_code == 1
    assert 'cannot read the PDF page tree' in failed.stderr
