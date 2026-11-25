"""Recover figure curves offline, through a stand-in for figrecover."""

from __future__ import annotations

import csv
import json
import types
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from click.testing import CliRunner

import paperminertoolkit.corpus.database as corpus
from paperminertoolkit import cli
from paperminertoolkit.workflows import digitize


class Calibration:
    """Map pixels to data by halving them."""

    origin = (0.0, 0.0)

    def to_data(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return the data coordinates of pixel positions."""
        return np.asarray(points, dtype=float) / 2


class Series:
    """A recovered curve, as figrecover returns one."""

    def __init__(self, name: str, style: str, points: Sequence[tuple[float, float]]) -> None:
        """Store the curve's name, style and pixel positions."""
        self.name = name
        self.style = style
        self.points = np.asarray(points, dtype=float)


class Panel:
    """A plot of a figure, as figrecover's read_panels returns one."""

    def __init__(self, label: str | None, offset: tuple[float, float] = (0.0, 0.0),
                 inset_of: Panel | None = None) -> None:
        """Give the panel an image, axes and position."""
        self.label = label
        self.offset = offset
        self.inset_of = inset_of
        self.image = np.zeros((10, 10, 3), dtype=np.uint8)
        self.axes = types.SimpleNamespace(calibration=Calibration(), x_title='Time (d)', y_title='Mass loss (%)')


def stand_in(panels: Sequence[Panel], error: Exception | None = None) -> types.ModuleType:
    """Build a figrecover stand-in that finds two curves in every panel."""
    module = types.ModuleType('figrecover')

    def read_panels(image: npt.NDArray[np.uint8]) -> list[Panel]:
        """Return the panels, or fail as an unreadable plot does."""
        if error is not None:
            raise error
        return list(panels)

    module.load_image = lambda path: np.full((20, 20, 3), 255, dtype=np.uint8)
    module.level_image = lambda image: (image, None)
    module.read_panels = read_panels
    module.find_series = lambda image, origin: [
        Series('1f77b4-solid', 'solid', [(2, 4), (4, 8)]),
        Series('ff7f0e-markers', 'markers', [(6, 2)]),
    ]
    return module


@pytest.fixture
def install(monkeypatch: pytest.MonkeyPatch) -> Callable[..., types.ModuleType]:
    """Return a function that puts a figrecover stand-in in place for one test."""
    def setup(panels: Sequence[Panel], error: Exception | None = None) -> types.ModuleType:
        """Install the stand-in and return it."""
        module = stand_in(panels, error)
        monkeypatch.setattr(digitize, '_figrecover', lambda: module)
        return module
    return setup


def add_figure(conn: Any, paper_id: str, figure_id: str, label: str, source: str = 'pubmed') -> None:
    """Store one figure image for a paper."""
    corpus.add_figure_asset(
        conn, {'paper_id': paper_id}, b'\x89PNG\r\n\x1a\n' + figure_id.encode(),
        figure_id=figure_id, caption='Mass loss of the films in compost.', source=source,
        source_url=f'https://example.org/{figure_id}.png', mime_type='image/png',
        original_filename=f'{figure_id}.png', metadata={'figure_label': label},
    )


@pytest.fixture
def corpus_db(tmp_path: Path) -> Path:
    """Two papers: one with figures 3 (twice, from two sources) and 4, one with none."""
    db_path = tmp_path / 'papers.db'
    with corpus.connect(db_path) as conn:
        corpus.upsert_paper(conn, {'paper_id': 'paper-1', 'doi': '10.1/one', 'title': 'One'})
        corpus.upsert_paper(conn, {'paper_id': 'paper-2', 'doi': '10.1/two', 'title': 'Two'})
        add_figure(conn, 'paper-1', 'pdf-figure-3', 'Figure 3', source='pdf-layout')
        add_figure(conn, 'paper-1', 'fig3', 'Figure 3')
        add_figure(conn, 'paper-1', 'fig4', 'Fig. 4')
    return db_path


@pytest.fixture
def records(tmp_path: Path) -> Path:
    """Scraped records naming figures in the ways models write them."""
    rows = [
        ('paper-1', 'PLA film', 'Figure 3a', 'None'),
        ('paper-1', 'PBS film', 'Fig. 3b', 'None'),
        ('paper-1', 'PCL film', 'Figure 4', '5% at 7 d'),
        ('paper-1', 'PHB film', 'Figure S2', 'None'),
        ('paper-1', 'PET film', 'Figure 9', 'None'),
        ('paper-2', 'PBAT film', 'Figure 1', 'None'),
        ('paper-9', 'Ghost film', 'Figure 1', 'None'),
        ('paper-1', 'Blank film', 'None', 'None'),
    ]
    frame = pd.DataFrame(rows, columns=['Paper id', 'Name', 'Mass loss figure', 'Mass loss curve'])
    path = tmp_path / 'scraped.csv'
    frame.to_csv(path)
    return path


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read the digitized CSV."""
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize(('text', 'expected'), [
    ('Figure 3', [('3', '', False)]),
    ('Fig. 4a', [('4', 'a', False)]),
    ('Figure 4(b)', [('4', 'b', False)]),
    ('Figs. 2 and 3', [('2', '', False), ('3', '', False)]),
    ('Figures 2-4', [('2', '', False), ('3', '', False), ('4', '', False)]),
    ('Figure 5a-c', [('5', 'abc', False)]),
    ('Figure 7a, b', [('7', 'ab', False)]),
    ('Figure 7 a and b', [('7', 'ab', False)]),
    ('Figure 2 (a)–(c)', [('2', 'abc', False)]),
    ('Fig. 2; Fig. 3c', [('2', '', False), ('3', 'c', False)]),
    ('Figure 3a and 3b', [('3', 'a', False), ('3', 'b', False)]),
    ('Figure S4', [('4', '', True)]),
    ('Supplementary Figure 2', [('2', '', True)]),
    ('Figure 2 and Table 3', [('2', '', False)]),
    ('Figure 3 of ref. 12', [('3', '', False)]),
    ('FIG. 10', [('10', '', False)]),
    ('Table 2', []),
    ('None', []),
    (float('nan'), []),
])
def test_figure_references_are_read_as_models_write_them(text: object, expected: list[tuple[str, str, bool]]) -> None:
    """Read figure numbers, panels, ranges and supplementary figures."""
    targets = digitize.parse_figure_reference(text)
    assert [(target.number, ''.join(target.panels), target.supplementary) for target in targets] == expected


def test_stored_figures_are_numbered_from_label_or_identifier() -> None:
    """Number a figure by its label, falling back to its identifier."""
    assert digitize._figure_number({'metadata': {'figure_label': 'Fig. 3.'}}) == ('3', False)
    assert digitize._figure_number({'metadata': {'figure_label': 'Figure S1'}}) == ('1', True)
    assert digitize._figure_number({'metadata': {'figure_id': 'polym-13-01855-f003'}}) == ('3', False)
    assert digitize._figure_number({'metadata': {'figure_id': 'graphical-abstract'}}) is None


def test_requests_group_references_by_paper_and_figure(records: Path) -> None:
    """Merge the records citing one figure, and honour the missing-value filter."""
    frame = pd.read_csv(records, index_col=0)

    requests = digitize.figure_requests(frame, 'Mass loss figure', missing_field='Mass loss curve')

    keys = [(request.paper_id, request.target.number, request.target.supplementary) for request in requests]
    assert keys == [('paper-1', '3', False), ('paper-1', '2', True), ('paper-1', '9', False),
                    ('paper-2', '1', False), ('paper-9', '1', False)]
    figure_3 = requests[0]
    assert figure_3.panels == {'a', 'b'} and not figure_3.whole
    assert figure_3.rows == ['0', '1']
    assert figure_3.names == ['PLA film', 'PBS film']
    assert figure_3.references == ['Figure 3a', 'Fig. 3b']
    every = digitize.figure_requests(frame, 'Mass loss figure')
    assert ('paper-1', '4') in [(request.paper_id, request.target.number) for request in every]
    with pytest.raises(KeyError, match='Figure field'):
        digitize.figure_requests(frame, 'Figure field')


def test_digitize_writes_points_checkpoints_and_resumes(
    corpus_db: Path, records: Path, tmp_path: Path, install: Callable[..., types.ModuleType],
) -> None:
    """Recover the cited panels once, skip them on a rerun, and redo them when forced."""
    install([Panel('a'), Panel('b', offset=(10.0, 0.0)), Panel('c')])
    output = tmp_path / 'digitized.csv'

    summary = digitize.digitize_figures(corpus_db, records, 'Mass loss figure', output,
                                        missing_field='Mass loss curve')

    assert {key: summary[key] for key in ('requests', 'digitized', 'skipped', 'failed', 'not_found',
                                          'supplementary', 'panels', 'series', 'points')} == {
        'requests': 5, 'digitized': 1, 'skipped': 0, 'failed': 0, 'not_found': 3,
        'supplementary': 1, 'panels': 2, 'series': 4, 'points': 6,
    }
    rows = read_rows(output)
    assert list(rows[0]) == digitize.OUTPUT_COLUMNS
    assert {row['Panel'] for row in rows} == {'a', 'b'}
    first = rows[0]
    assert first['Figure id'] == 'fig3'  # the publisher's figure, not the PDF-detected copy
    assert first['Figure label'] == 'Figure 3'
    assert first['Referenced as'] == 'Figure 3a; Fig. 3b'
    assert first['Record rows'] == '0; 1'
    assert first['Record names'] == 'PLA film; PBS film'
    assert (first['Series'], first['Series style'], first['x'], first['y']) == ('1f77b4-solid', 'solid', '1', '2')
    assert (first['X title'], first['Y title']) == ('Time (d)', 'Mass loss (%)')
    with corpus.connect(corpus_db) as conn:
        statuses = {asset['metadata']['figure_id']: asset['metadata'].get('digitization_status')
                    for asset in corpus.get_figure_assets(conn, 'paper-1')}
    assert statuses == {'fig3': 'succeeded', 'pdf-figure-3': None, 'fig4': None}

    again = digitize.digitize_figures(corpus_db, records, 'Mass loss figure', output,
                                      missing_field='Mass loss curve')
    assert (again['digitized'], again['skipped']) == (0, 1)
    assert len(read_rows(output)) == 6

    forced = digitize.digitize_figures(corpus_db, records, 'Mass loss figure', output,
                                       missing_field='Mass loss curve', force=True)
    assert forced['digitized'] == 1
    assert len(read_rows(output)) == 12


def test_unreadable_figures_are_marked_failed_with_the_reason(
    corpus_db: Path, records: Path, tmp_path: Path, install: Callable[..., types.ModuleType],
) -> None:
    """Record why figrecover could not read a plot, and write no points for it."""
    install([], error=ValueError('found no row of 3 or more numbers to read as x-axis ticks'))
    output = tmp_path / 'digitized.csv'

    summary = digitize.digitize_figures(corpus_db, records, 'Mass loss figure', output,
                                        missing_field='Mass loss curve')

    assert summary['failed'] == 1
    assert any('x-axis ticks' in message for message in summary['errors'])
    assert read_rows(output) == []
    with corpus.connect(corpus_db) as conn:
        figure = next(asset for asset in corpus.get_figure_assets(conn, 'paper-1')
                      if asset['metadata']['figure_id'] == 'fig3')
    assert figure['metadata']['digitization_status'] == 'failed'
    assert figure['metadata']['digitization_error'].startswith('ValueError: found no row')
    assert figure['metadata'].get('extraction_status') is None


def test_every_panel_is_kept_when_labels_cannot_be_read(
    corpus_db: Path, records: Path, tmp_path: Path, install: Callable[..., types.ModuleType],
) -> None:
    """Keep all panels, and say so, when the cited panel letters were not read."""
    outer = Panel(None)
    install([outer, Panel(None, inset_of=outer)])

    summary = digitize.digitize_figures(corpus_db, records, 'Mass loss figure', tmp_path / 'out.csv',
                                        missing_field='Mass loss curve')

    assert summary['panels'] == 2
    assert {row['Panel'] for row in read_rows(tmp_path / 'out.csv')} == {'1', 'inset'}
    assert any('panel labels could not be read' in message for message in summary['errors'])


def test_figures_are_detected_from_the_pdf_of_a_paper_with_none(
    corpus_db: Path, records: Path, tmp_path: Path, install: Callable[..., types.ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Detect a figure-less paper's figures from its PDF, unless detection is off."""
    install([Panel('a')])
    with corpus.connect(corpus_db) as conn:
        corpus.add_asset(conn, {'paper_id': 'paper-2'}, b'%PDF-1.7 stand-in', role='pdf', kind='pdf',
                         mime_type='application/pdf', source='pubmed', original_filename='two.pdf')
    detected = []

    def detect(conn: Any, paper: dict[str, Any], pdf_path: str) -> None:
        """Store figure 1 as a PDF-detected figure."""
        detected.append(Path(pdf_path).read_bytes())
        add_figure(conn, paper['paper_id'], 'pdf-figure-1', 'Figure 1', source='pdf-layout')

    monkeypatch.setattr(digitize, 'store_pdf_layout_figures', detect)

    off = digitize.digitize_figures(corpus_db, records, 'Mass loss figure', tmp_path / 'off.csv',
                                    missing_field='Mass loss curve', detect=False)
    assert off['not_found'] == 3 and detected == []

    on = digitize.digitize_figures(corpus_db, records, 'Mass loss figure', tmp_path / 'on.csv',
                                   missing_field='Mass loss curve')
    assert detected == [b'%PDF-1.7 stand-in']
    assert (on['not_found'], on['skipped'], on['digitized']) == (2, 1, 1)  # figure 3 was done by the first run
    assert {row['Figure id'] for row in read_rows(tmp_path / 'on.csv')} == {'pdf-figure-1'}


def test_limit_stops_after_the_given_number_of_figures(
    corpus_db: Path, records: Path, tmp_path: Path, install: Callable[..., types.ModuleType],
) -> None:
    """Attempt no more figures than the limit."""
    install([Panel('a')])

    summary = digitize.digitize_figures(corpus_db, records, 'Mass loss figure', tmp_path / 'out.csv', limit=1)

    assert summary['digitized'] + summary['failed'] == 1


def test_missing_figrecover_explains_how_to_install_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """Name the install command when figrecover or its OCR extra is absent."""
    def refuse(name: str) -> types.ModuleType:
        """Fail to import optional packages."""
        raise ImportError(f'No module named {name!r}')

    monkeypatch.setattr(digitize.importlib, 'import_module', refuse)
    with pytest.raises(RuntimeError, match='figrecover with its OCR extra'):
        digitize._figrecover()


def test_cli_reports_json_and_rejects_missing_columns(
    corpus_db: Path, records: Path, tmp_path: Path, install: Callable[..., types.ModuleType],
) -> None:
    """Run the command end to end, and turn a missing column into a usage error."""
    install([Panel('a'), Panel('b')])
    output = tmp_path / 'digitized.csv'
    runner = CliRunner()

    result = runner.invoke(cli.main, [
        'digitize', str(corpus_db), str(records), '--figure-field', 'Mass loss figure',
        '--only-missing', 'Mass loss curve', '--output', str(output), '--json',
    ])

    assert result.exit_code == 0, result.output
    summary = json.loads(result.stdout)
    assert summary['digitized'] == 1 and summary['points'] == 6
    text = runner.invoke(cli.main, [
        'digitize', str(corpus_db), str(records), '--figure-field', 'Mass loss figure',
        '--output', str(output), '--force',
    ])
    assert text.exit_code == 0, text.output
    assert 'Figures requested: 6; recovered 2' in text.output
    missing = runner.invoke(cli.main, [
        'digitize', str(corpus_db), str(records), '--figure-field', 'Degradation figure',
    ])
    assert missing.exit_code == 1
    assert "column 'Degradation figure' is not in the records" in missing.output


def test_preview_draws_recovered_curves(tmp_path: Path, install: Callable[..., types.ModuleType]) -> None:
    """Save a preview image with the curves drawn over the figure."""
    pytest.importorskip('cv2')
    install([Panel('a')])
    image = tmp_path / 'figure.png'
    image.write_bytes(b'not read by the stand-in')

    panels, notes = digitize.digitize_image(image, ['a'], preview=tmp_path / 'previews' / 'figure.png')

    assert notes == []
    assert [len(panel.series) for panel in panels] == [2]
    assert (tmp_path / 'previews' / 'figure.png').stat().st_size > 0


def test_digitization_status_is_kept_apart_from_extraction_status(tmp_path: Path) -> None:
    """Store digitization progress in its own keys of the figure's metadata."""
    with corpus.connect(tmp_path / 'papers.db') as conn:
        corpus.upsert_paper(conn, {'paper_id': 'paper-1'})
        add_figure(conn, 'paper-1', 'fig1', 'Figure 1')
        assert corpus.set_figure_extraction_status(conn, 'paper-1', 'fig1', 'succeeded')
        assert corpus.set_figure_digitization_status(conn, 'paper-1', 'fig1', 'failed', error=' no axes ')
        assert not corpus.set_figure_digitization_status(conn, 'paper-1', 'fig9', 'failed')
        metadata = corpus.get_figure_assets(conn, 'paper-1')[0]['metadata']
        with pytest.raises(ValueError, match='status'):
            corpus.set_figure_digitization_status(conn, 'paper-1', 'fig1', ' ')
    assert metadata['extraction_status'] == 'succeeded'
    assert (metadata['digitization_status'], metadata['digitization_error']) == ('failed', 'no axes')
