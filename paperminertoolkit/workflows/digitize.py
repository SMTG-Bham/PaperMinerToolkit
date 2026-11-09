"""Recover the plotted data of figures that extracted records name.

A recipe can ask for the figure that plots a result, such as the
``Mass loss figure`` of ``polymer_degradation``, when the paper gives the
values only as a curve. This stage reads those references from a scraped or
stored CSV, finds each figure in the corpus, and recovers its curves with
figrecover (https://github.com/SMTG-Bham/figrecover). figrecover reads each
panel's axes from its tick labels by OCR and tells curves apart by colour,
line style and markers; it does not read legends, so a curve is named by how
it is drawn, such as ``1f77b4-solid``, and matching it to a sample is left to
the reader, helped by the figure's caption and the records that cite it.

Figures come from the corpus as ``pmt scrape --image-extraction layout`` uses
them: stored structured figures first, and for a paper with none, figures
detected from its PDF and stored for later runs. Progress is checkpointed per
figure in the figure asset's metadata, so a rerun skips figures already
recovered; ``force`` recovers them again.

figrecover is optional and is not published on PyPI. Install it with its OCR
extra::

    pip install "figrecover[ocr] @ git+https://github.com/SMTG-Bham/figrecover"
"""

from __future__ import annotations

import csv
import importlib
import os
import re
import shutil
import sqlite3
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd

from paperminertoolkit.corpus.database import (
    connect,
    get_asset,
    get_figure_assets,
    paper_rows,
    set_figure_digitization_status,
)
from paperminertoolkit.workflows.figures import PDF_LAYOUT_SOURCE, store_pdf_layout_figures

FIGRECOVER_INSTALL = 'pip install "figrecover[ocr] @ git+https://github.com/SMTG-Bham/figrecover"'
OUTPUT_COLUMNS = [
    'Paper id', 'doi', 'Figure id', 'Figure label', 'Figure source', 'Referenced as',
    'Record rows', 'Record names', 'Panel', 'Series', 'Series style', 'X title', 'Y title', 'x', 'y',
]
_MISSING = {'', 'none', 'nan', 'null', 'n/a', 'not reported'}
_KEYWORD = re.compile(r'\b(?:fig(?:ure)?s?)\b\.?', re.IGNORECASE)
_STOP = re.compile(r'\b(?:table|scheme|eq(?:uation)?)s?\b|;', re.IGNORECASE)
_LETTER = r'(?<![a-z])[a-l](?![a-z])'
_PANELS = rf'\(?{_LETTER}\)?(?:\s*(?:,|and|&|-|–|to)\s*\(?{_LETTER}\)?)*'
_SUPPLEMENTARY = re.compile(r'\b(?:supplementary|supporting|SI)\s*$', re.IGNORECASE)
_TOKEN = re.compile(rf'(?P<supplementary>S)?(?P<number>\d+)(?P<panels>\s*{_PANELS})?', re.IGNORECASE)
_RANGE = re.compile(r'^\s*(?:-|–|to)\s*$')


@dataclass(frozen=True)
class FigureTarget:
    """One figure, and the panels of it, that a reference names.

    Parameters
    ----------
    number : str
        Figure number without leading zeros, such as ``"4"``.
    panels : tuple[str, ...]
        Lower-case panel letters, or empty for the whole figure.
    supplementary : bool
        Whether the reference is to a supporting-information figure (``S4``).
    """

    number: str
    panels: tuple[str, ...] = ()
    supplementary: bool = False


def _panel_letters(spec: str) -> tuple[str, ...]:
    """Expand a panel specification such as ``a-c`` or ``(a), (b)`` into letters.

    Parameters
    ----------
    spec : str
        Text following a figure number.

    Returns
    -------
    tuple[str, ...]
        Lower-case letters in order, without repeats.
    """
    spec = spec.lower()
    letters = list(re.finditer(_LETTER, spec))
    expanded: list[str] = []
    for index, match in enumerate(letters):
        if index and _RANGE.match(re.sub(r'[()]', '', spec[letters[index - 1].end():match.start()])):
            start = letters[index - 1].group()
            expanded.extend(chr(code) for code in range(ord(start) + 1, ord(match.group())))
        expanded.append(match.group())
    return tuple(dict.fromkeys(expanded))


def parse_figure_reference(text: object) -> list[FigureTarget]:
    """Read the figures and panels that a record's figure field names.

    Understands the forms papers and models write: ``Figure 4``, ``Fig. 4a``,
    ``Figure 4(b)``, ``Figs. 2 and 3``, ``Figures 2-4``, ``Figure 5a-c``,
    ``Figure 7a, b``, ``Fig. 2; Fig. 3c``, and supplementary figures written
    ``Figure S4`` or ``Supplementary Figure 4``. Tables, schemes and missing
    values name no figure.

    Parameters
    ----------
    text : object
        Field value; non-strings and missing markers give no figures.

    Returns
    -------
    list[FigureTarget]
        Figures in the order they are named, without repeats.
    """
    if not isinstance(text, str) or text.strip().casefold() in _MISSING:
        return []
    targets: list[FigureTarget] = []
    keywords = list(_KEYWORD.finditer(text))
    for index, keyword in enumerate(keywords):
        end = keywords[index + 1].start() if index + 1 < len(keywords) else len(text)
        segment = text[keyword.end():end]
        stop = _STOP.search(segment)
        if stop:
            segment = segment[:stop.start()]
        in_supplement = bool(_SUPPLEMENTARY.search(text[:keyword.start()]))
        previous: re.Match[str] | None = None
        for token in _TOKEN.finditer(segment):
            between = segment[previous.end():token.start()] if previous else segment[:token.start()]
            if previous is None and between.strip(' .:(#'):
                break
            if previous is not None and not re.fullmatch(r'\s*(?:,|and|&|-|–|to)?\s*', between):
                break
            supplementary = in_supplement or bool(token.group('supplementary'))
            number = str(int(token.group('number')))
            panels = _panel_letters(token.group('panels') or '')
            if (previous is not None and _RANGE.match(between) and not previous.group('panels')
                    and not panels and targets):
                start = int(targets[-1].number)
                for between_number in range(start + 1, int(number)):
                    targets.append(FigureTarget(str(between_number), (), supplementary))
            targets.append(FigureTarget(number, panels, supplementary))
            previous = token
    return list(dict.fromkeys(targets))


def _figure_number(asset: Mapping[str, Any]) -> tuple[str, bool] | None:
    """Return the number of a stored figure, and whether it is supplementary.

    Parameters
    ----------
    asset : Mapping[str, Any]
        Figure asset from the corpus.

    Returns
    -------
    tuple[str, bool] or None
        Number and supplementary flag read from the figure's label, or from
        its identifier (``fig3``, ``F3``) when the label has none.
    """
    metadata = asset.get('metadata') or {}
    targets = parse_figure_reference(str(metadata.get('figure_label') or ''))
    if targets:
        return targets[0].number, targets[0].supplementary
    match = re.search(r'(?:^|[^a-z])f(?:ig(?:ure)?)?[^a-z0-9]*(s?)0*(\d+)$',
                      str(metadata.get('figure_id') or ''), re.IGNORECASE)
    if match:
        return str(int(match.group(2))), bool(match.group(1))
    return None


@dataclass
class _Request:
    """The references in a CSV to one figure of one paper.

    Parameters
    ----------
    paper_id : str
        Corpus paper identifier.
    target : FigureTarget
        The figure, panels cleared; panels wanted are collected separately.
    panels : set[str]
        Panels the references name; empty when any names the whole figure.
    whole : bool
        Whether any reference names the whole figure.
    references : list[str]
        Distinct reference texts.
    rows : list[str]
        CSV row labels of the citing records.
    names : list[str]
        ``Name`` values of the citing records, when the CSV has that column.
    """

    paper_id: str
    target: FigureTarget
    panels: set[str] = field(default_factory=set)
    whole: bool = False
    references: list[str] = field(default_factory=list)
    rows: list[str] = field(default_factory=list)
    names: list[str] = field(default_factory=list)


def _is_missing(value: object) -> bool:
    """Return whether a CSV cell holds no value.

    Parameters
    ----------
    value : object
        Cell value read by pandas.

    Returns
    -------
    bool
        True for NaN, empty text and the recipe missing markers.
    """
    if value is None or (isinstance(value, float) and value != value):
        return True
    return str(value).strip().casefold() in _MISSING


def figure_requests(
    records: pd.DataFrame,
    figure_field: str,
    *,
    missing_field: str | None = None,
) -> list[_Request]:
    """Group a CSV's figure references by paper and figure.

    Parameters
    ----------
    records : pandas.DataFrame
        Scraped or stored records with a ``Paper id`` column.
    figure_field : str
        Column naming the figure that plots a record's result.
    missing_field : str or None, optional
        Only use records whose value in this column is missing, such as the
        curve a figure should supply.

    Returns
    -------
    list[_Request]
        One request per paper and figure, in the order first cited.

    Raises
    ------
    KeyError
        If a required column is absent.
    """
    for column in ('Paper id', figure_field, *([missing_field] if missing_field else [])):
        if column not in records.columns:
            raise KeyError(f'column {column!r} is not in the records')
    requests: dict[tuple[str, FigureTarget], _Request] = {}
    for label, record in records.iterrows():
        if missing_field and not _is_missing(record[missing_field]):
            continue
        paper_id = str(record['Paper id'] or '').strip()
        if not paper_id or _is_missing(paper_id):
            continue
        reference = str(record[figure_field])
        for target in parse_figure_reference(reference):
            key = (paper_id, FigureTarget(target.number, (), target.supplementary))
            request = requests.setdefault(key, _Request(paper_id, key[1]))
            if target.panels:
                request.panels.update(target.panels)
            else:
                request.whole = True
            if reference.strip() not in request.references:
                request.references.append(reference.strip())
            request.rows.append(str(label))
            name = record.get('Name') if 'Name' in records.columns else None
            if not _is_missing(name) and str(name) not in request.names:
                request.names.append(str(name))
    return list(requests.values())


def _figrecover() -> ModuleType:
    """Import figrecover, explaining how to install it when it is missing.

    Returns
    -------
    types.ModuleType
        The figrecover package.

    Raises
    ------
    RuntimeError
        If figrecover or its OCR extra is not installed.
    """
    try:
        module = importlib.import_module('figrecover')
        # figrecover reads axes by OCR only when its ocr extra is installed.
        importlib.import_module('rapidocr')
    except ImportError as error:
        raise RuntimeError(
            f'pmt digitize needs figrecover with its OCR extra, which is not on PyPI: {FIGRECOVER_INSTALL}'
        ) from error
    return module


@dataclass
class _PanelData:
    """The curves recovered from one panel of a figure.

    Parameters
    ----------
    name : str
        Panel letter, ``inset`` for an inset, or its position from 1.
    x_title, y_title : str
        Axis titles as read, or ``x`` and ``y``.
    series : list[tuple[str, str, list[tuple[float, float]], list[tuple[float, float]]]]
        Per curve: its name, style, data points, and the pixel positions in
        the whole image used to draw a preview.
    """

    name: str
    x_title: str
    y_title: str
    series: list[tuple[str, str, list[tuple[float, float]], list[tuple[float, float]]]]


def digitize_image(
    path: str | PathLike[str],
    panels: Iterable[str] = (),
    *,
    preview: str | PathLike[str] | None = None,
) -> tuple[list[_PanelData], list[str]]:
    """Recover every curve in a figure image with figrecover.

    The image is levelled, split into panels whose axes are read from their
    tick labels, and each panel's curves are traced and mapped to data.

    Parameters
    ----------
    path : str or os.PathLike[str]
        Figure image.
    panels : Iterable[str], optional
        Panel letters to keep. When figrecover reads no panel labels, every
        panel is kept and a note says so.
    preview : str or os.PathLike[str] or None, optional
        Where to save a faded copy of the figure with the recovered curves.

    Returns
    -------
    tuple[list[_PanelData], list[str]]
        The recovered panels, and notes on what was not as asked.

    Raises
    ------
    RuntimeError
        If figrecover is not installed.
    ValueError
        If no panel's axes could be read.
    """
    figrecover = _figrecover()
    image = figrecover.load_image(path)
    image, _ = figrecover.level_image(image)
    found = list(figrecover.read_panels(image))
    wanted = {panel.lower() for panel in panels}
    notes: list[str] = []
    labelled = {str(panel.label).lower() for panel in found if panel.label}
    if wanted and labelled:
        missing = sorted(wanted - labelled)
        if missing:
            notes.append(f'panel(s) {", ".join(missing)} not found; read {", ".join(sorted(labelled))}')
        found = [panel for panel in found
                 if (panel.label and panel.label.lower() in wanted)
                 or (panel.inset_of is not None and panel.inset_of.label
                     and panel.inset_of.label.lower() in wanted)]
    elif wanted and len(found) > 1:
        notes.append('panel labels could not be read, so every panel was kept')
    height = image.shape[0]
    results = []
    for index, panel in enumerate(found, 1):
        calibration = panel.axes.calibration
        name = (panel.label.lower() if panel.label else
                'inset' if panel.inset_of is not None else str(index))
        series = []
        for curve in figrecover.find_series(panel.image, calibration.origin):
            points = calibration.to_data(curve.points)
            pixels = [(float(x) + panel.offset[0], height - (float(y) + panel.offset[1]))
                      for x, y in curve.points]
            series.append((curve.name, str(curve.style),
                           [(float(x), float(y)) for x, y in points], pixels))
        results.append(_PanelData(name, panel.axes.x_title or 'x', panel.axes.y_title or 'y', series))
    if preview is not None:
        _save_preview(image, results, preview)
    return results, notes


def _save_preview(image: Any, panels: Sequence[_PanelData], path: str | PathLike[str]) -> None:
    """Draw the recovered curves over a faded copy of the figure.

    Parameters
    ----------
    image : numpy.ndarray
        Levelled RGB figure image.
    panels : Sequence[_PanelData]
        Recovered panels, with pixel positions in the whole image.
    path : str or os.PathLike[str]
        Output image path.
    """
    cv2 = importlib.import_module('cv2')
    faded = (image.astype('float32') * 0.35 + 255 * 0.65).astype('uint8')
    canvas = cv2.cvtColor(faded, cv2.COLOR_RGB2BGR)
    for panel in panels:
        for name, _, _, pixels in panel.series:
            # Series are named after their colour in hex RGB; OpenCV draws in BGR.
            colour = ((int(name[4:6], 16), int(name[2:4], 16), int(name[0:2], 16))
                      if re.match(r'[0-9a-f]{6}', name) else (0, 0, 255))
            for x, y in pixels:
                cv2.circle(canvas, (int(round(x)), int(round(y))), 1, colour, -1)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def _choose_asset(assets: Sequence[Mapping[str, Any]], target: FigureTarget) -> Mapping[str, Any] | None:
    """Pick the stored figure a target names, preferring publisher figures.

    Parameters
    ----------
    assets : Sequence[Mapping[str, Any]]
        The paper's figure assets.
    target : FigureTarget
        Figure to find.

    Returns
    -------
    Mapping[str, Any] or None
        The matching asset, a structured figure before a PDF-detected one.
    """
    matches = [asset for asset in assets
               if _figure_number(asset) == (target.number, target.supplementary)]
    matches.sort(key=lambda asset: asset.get('source') == PDF_LAYOUT_SOURCE)
    return matches[0] if matches else None


def _safe_name(value: str) -> str:
    """Make a value safe as part of a file name.

    Parameters
    ----------
    value : str
        Paper or figure identifier.

    Returns
    -------
    str
        Letters, digits, dots, dashes and underscores only.
    """
    return re.sub(r'[^A-Za-z0-9._-]+', '_', value).strip('_') or 'figure'


def digitize_figures(
    db_path: str | PathLike[str],
    records_path: str | PathLike[str],
    figure_field: str,
    output_path: str | PathLike[str],
    *,
    missing_field: str | None = None,
    preview_dir: str | PathLike[str] | None = None,
    detect: bool = True,
    force: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Recover the curves of the figures that records name, and write them as CSV.

    Parameters
    ----------
    db_path : str or os.PathLike[str]
        Corpus database.
    records_path : str or os.PathLike[str]
        Scraped or stored records CSV, with a ``Paper id`` column.
    figure_field : str
        Column naming each record's figure, such as ``Mass loss figure``.
    output_path : str or os.PathLike[str]
        CSV to append the recovered points to, one row per point, with the
        columns of :data:`OUTPUT_COLUMNS`.
    missing_field : str or None, optional
        Only follow records whose value in this column is missing.
    preview_dir : str or os.PathLike[str] or None, optional
        Directory for one preview image per recovered figure.
    detect : bool, default=True
        Detect and store figures from a paper's PDF when the corpus holds
        none for it.
    force : bool, default=False
        Recover figures again even when already recovered.
    limit : int or None, optional
        Stop after this many figures have been attempted.

    Returns
    -------
    dict[str, Any]
        Counts of requests and outcomes, the output path, and up to 20
        failure messages.

    Raises
    ------
    RuntimeError
        If figrecover is not installed.
    KeyError
        If the records lack a required column.
    """
    _figrecover()
    records = pd.read_csv(records_path, index_col=0, low_memory=False)
    requests = figure_requests(records, figure_field, missing_field=missing_field)
    summary: dict[str, Any] = {
        'requests': len(requests), 'supplementary': 0, 'not_found': 0, 'skipped': 0,
        'digitized': 0, 'failed': 0, 'panels': 0, 'series': 0, 'points': 0,
        'output': str(output_path), 'errors': [],
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    new_file = not output.exists() or output.stat().st_size == 0
    attempted = 0
    by_paper: dict[str, list[_Request]] = {}
    for request in requests:
        by_paper.setdefault(request.paper_id, []).append(request)
    conn = connect(db_path)
    temp_dir = tempfile.mkdtemp(prefix='pmt-digitize-')
    try:
        papers = {paper['paper_id']: paper for paper in paper_rows(conn)}
        with output.open('a', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
            if new_file:
                writer.writeheader()
            for paper_id, paper_requests in by_paper.items():
                if limit is not None and attempted >= limit:
                    break
                paper = papers.get(paper_id)
                if paper is None:
                    summary['not_found'] += len(paper_requests)
                    _note(summary, f'{paper_id}: not in the corpus')
                    continue
                assets = _paper_figures(conn, paper, temp_dir, detect=detect, summary=summary)
                for request in paper_requests:
                    if limit is not None and attempted >= limit:
                        break
                    label = f'{paper_id} Figure {"S" if request.target.supplementary else ""}{request.target.number}'
                    asset = _choose_asset(assets, request.target)
                    if asset is None:
                        key = 'supplementary' if request.target.supplementary else 'not_found'
                        summary[key] += 1
                        if not request.target.supplementary:
                            _note(summary, f'{label}: no stored or detected figure with that number')
                        continue
                    metadata = asset.get('metadata') or {}
                    figure_id = str(metadata.get('figure_id') or '')
                    if not force and metadata.get('digitization_status') == 'succeeded':
                        summary['skipped'] += 1
                        continue
                    attempted += 1
                    image_path = os.path.join(temp_dir, f'{_safe_name(paper_id)}-{_safe_name(figure_id)}.img')
                    Path(image_path).write_bytes(asset['content'])
                    preview = (Path(preview_dir) / f'{_safe_name(paper_id)}-{_safe_name(figure_id)}.png'
                               if preview_dir else None)
                    wanted = () if request.whole else sorted(request.panels)
                    try:
                        panels, notes = digitize_image(image_path, wanted, preview=preview)
                    except Exception as error:  # figrecover raises several kinds for unreadable plots
                        message = f'{type(error).__name__}: {error}'
                        set_figure_digitization_status(conn, paper_id, figure_id, 'failed', error=message)
                        summary['failed'] += 1
                        _note(summary, f'{label}: {message}')
                        continue
                    rows = _output_rows(paper, asset, request, panels)
                    writer.writerows(rows)
                    handle.flush()
                    summary['digitized'] += 1
                    summary['panels'] += len(panels)
                    summary['series'] += sum(len(panel.series) for panel in panels)
                    summary['points'] += len(rows)
                    status = 'succeeded' if rows else 'failed'
                    error = '; '.join(notes) if rows else '; '.join([*notes, 'no curves found'])
                    set_figure_digitization_status(conn, paper_id, figure_id, status, error=error)
                    if notes or not rows:
                        _note(summary, f'{label}: {error}')
    finally:
        conn.close()
        shutil.rmtree(temp_dir, ignore_errors=True)
    return summary


def _note(summary: dict[str, Any], message: str) -> None:
    """Keep the first 20 failure messages and echo each to stderr.

    Parameters
    ----------
    summary : dict[str, Any]
        Run summary holding the ``errors`` list.
    message : str
        Message to record.
    """
    print(f'pmt digitize: {message}', file=sys.stderr)
    if len(summary['errors']) < 20:
        summary['errors'].append(message)


def _paper_figures(
    conn: sqlite3.Connection,
    paper: Mapping[str, Any],
    temp_dir: str,
    *,
    detect: bool,
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    """Load a paper's figure assets, detecting them from its PDF if it has none.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open corpus connection.
    paper : Mapping[str, Any]
        Corpus paper row.
    temp_dir : str
        Directory for the PDF copy that detection reads.
    detect : bool
        Whether detection may run.
    summary : dict[str, Any]
        Run summary for failure messages.

    Returns
    -------
    list[dict[str, Any]]
        Figure assets with their image bytes.
    """
    paper_id = paper['paper_id']
    assets = get_figure_assets(conn, paper_id, include_content=True)
    if assets or not detect:
        return assets
    pdf = get_asset(conn, paper_id, 'pdf')
    if pdf is None:
        return []
    pdf_path = os.path.join(temp_dir, f'{_safe_name(paper_id)}.pdf')
    Path(pdf_path).write_bytes(pdf['content'])
    try:
        store_pdf_layout_figures(conn, paper, pdf_path)
    except (OSError, RuntimeError, ValueError) as error:
        _note(summary, f'{paper_id}: figure detection failed: {error}')
        return []
    return get_figure_assets(conn, paper_id, include_content=True)


def _output_rows(
    paper: Mapping[str, Any],
    asset: Mapping[str, Any],
    request: _Request,
    panels: Sequence[_PanelData],
) -> list[dict[str, Any]]:
    """Flatten recovered panels into one CSV row per point.

    Parameters
    ----------
    paper : Mapping[str, Any]
        Corpus paper row.
    asset : Mapping[str, Any]
        The figure asset digitized.
    request : _Request
        The references that asked for the figure.
    panels : Sequence[_PanelData]
        Recovered panels.

    Returns
    -------
    list[dict[str, Any]]
        Rows keyed by :data:`OUTPUT_COLUMNS`.
    """
    metadata = asset.get('metadata') or {}
    common = {
        'Paper id': paper['paper_id'],
        'doi': paper.get('doi') or '',
        'Figure id': metadata.get('figure_id') or '',
        'Figure label': metadata.get('figure_label') or '',
        'Figure source': asset.get('source') or '',
        'Referenced as': '; '.join(request.references),
        'Record rows': '; '.join(request.rows),
        'Record names': '; '.join(request.names),
    }
    rows = []
    for panel in panels:
        for name, style, points, _ in panel.series:
            for x, y in points:
                rows.append(common | {
                    'Panel': panel.name, 'Series': name, 'Series style': style,
                    'X title': panel.x_title, 'Y title': panel.y_title,
                    'x': f'{x:.6g}', 'y': f'{y:.6g}',
                })
    return rows
