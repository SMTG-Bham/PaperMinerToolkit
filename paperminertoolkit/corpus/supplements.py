"""Store supplementary files and their independent extraction checkpoints."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable, Mapping, MutableMapping
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from paperminertoolkit.corpus.database import (
    _asset_dict, link_asset, normalize_paper, store_blob, upsert_paper,
)

SUPPLEMENT_ROLE = 'supplement'


def get_supplements(
    conn: sqlite3.Connection,
    paper_id: str,
    include_content: bool = True,
) -> list[dict[str, Any]]:
    """Load every supplement for a paper in deterministic acquisition order.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open corpus connection.
    paper_id : str
        Canonical identifier of the owning paper.
    include_content : bool, default=True
        Include decompressed bytes; disable for inventory and resume checks.

    Returns
    -------
    list[dict[str, Any]]
        Assets with provenance and extraction state in ``metadata``.
    """
    content_columns = ', b.compression, b.content' if include_content else ''
    rows = conn.execute(
        f'''SELECT p.paper_id, p.doi, p.title, a.role, a.source,
                   a.original_filename, a.metadata_json, a.created_at,
                   b.blob_id, b.kind, b.mime_type, b.original_size, b.stored_size
                   {content_columns}
            FROM paper_assets a JOIN papers p ON p.paper_id = a.paper_id
            JOIN blobs b ON b.blob_id = a.blob_id
            WHERE a.paper_id = ? AND a.role = ?
            ORDER BY a.created_at, a.source, a.original_filename''',
        (paper_id, SUPPLEMENT_ROLE),
    ).fetchall()
    return [_asset_dict(row, include_content=include_content) for row in rows]


def add_supplement(
    conn: sqlite3.Connection,
    paper: Mapping[str, Any],
    content: str | bytes | Path | Iterable[int],
    *,
    source: str = '',
    source_url: str = '',
    original_filename: str = '',
    mime_type: str = 'application/octet-stream',
    metadata: Mapping[str, Any] | None = None,
) -> str:
    """Attach a supplementary file without replacing the main article.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open corpus connection.
    paper : Mapping[str, Any]
        Owning paper metadata. Mutable inputs also receive updated SI state.
    content : str, bytes, pathlib.Path, or Iterable[int]
        File contents or a local file to read.
    source : str, optional
        Acquisition provider.
    source_url : str, optional
        Original download URL; also accepted inside ``metadata``.
    original_filename : str, optional
        Original name. Distinct URLs sharing a name receive a stable suffix.
    mime_type : str, default='application/octet-stream'
        Media type of the uncompressed content.
    metadata : Mapping[str, Any] or None, optional
        Caption, license, source identifiers, and other provenance.

    Returns
    -------
    str
        Content-addressed blob identifier. Replacing changed bytes resets only
        the affected supplement's checkpoints; identical bytes preserve them.
    """
    details = dict(metadata or {})
    source_url = source_url or str(details.get('source_url') or '')
    details['source_url'] = source_url
    paper_id = normalize_paper(paper)['paper_id']
    existing_row = conn.execute('SELECT * FROM papers WHERE paper_id = ?', (paper_id,)).fetchone()
    merged_paper = dict(existing_row) if existing_row else {}
    merged_paper.update(paper)
    upsert_paper(conn, merged_paper)
    assets = get_supplements(conn, paper_id, include_content=False)
    filename = original_filename or unquote(urlsplit(source_url).path.rsplit('/', 1)[-1])
    filename = filename.replace('\\', '/').rsplit('/', 1)[-1] or 'supplement'
    matching = next((asset for asset in assets if asset['source'] == source and (
        (source_url and asset['metadata'].get('source_url') == source_url)
        or (not source_url and asset['original_filename'] == filename)
    )), None)
    if matching:
        filename = matching['original_filename']
    elif any(asset['source'] == source and asset['original_filename'] == filename
             for asset in assets):
        suffix = hashlib.sha256(source_url.encode()).hexdigest()[:12]
        path = Path(filename)
        filename = f'{path.stem}-{suffix}{path.suffix}'
    blob_id = store_blob(conn, content, kind=SUPPLEMENT_ROLE, mime_type=mime_type)
    unchanged = matching is not None and matching['blob_id'] == blob_id
    if unchanged:
        details = {**matching['metadata'], **details}
    else:
        for stage in ('text', 'image'):
            details[f'{stage}_extraction_status'] = 'pending'
            details[f'{stage}_extraction_error'] = ''
            details.pop(f'{stage}_extraction_members', None)
            merged_paper[f'supplement_{stage}_scrape_status'] = 'pending'
        merged_paper['store_status'] = 'pending'
        upsert_paper(conn, merged_paper)
        if isinstance(paper, MutableMapping):
            for key in ('supplement_text_scrape_status', 'supplement_image_scrape_status',
                        'store_status'):
                paper[key] = merged_paper[key]
    link_asset(conn, paper_id, blob_id, SUPPLEMENT_ROLE, source=source,
               original_filename=filename, metadata=details)
    return blob_id


def set_supplement_extraction_status(
    conn: sqlite3.Connection,
    paper_id: str,
    blob_id: str,
    stage: str,
    status: str,
    *,
    error: str = '',
    members: Any = None,
) -> bool:
    """Persist a supplement checkpoint independently of the article stages.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open corpus connection.
    paper_id : str
        Owning paper identifier.
    blob_id : str
        Content identifier; all matching links receive the checkpoint.
    stage : {'text', 'image'}
        Extraction stage.
    status : str
        Stage outcome, such as ``succeeded``, ``failed``, or ``unsupported``.
    error : str, optional
        Failure or unsupported-format detail.
    members : Any, optional
        JSON-serializable archive member outcomes or provenance.

    Returns
    -------
    bool
        Whether an existing link was updated.

    Raises
    ------
    ValueError
        If the stage is unsupported or status is empty.
    """
    if stage not in {'text', 'image'} or not status.strip():
        raise ValueError('stage must be text or image and status must not be empty')
    rows = conn.execute(
        'SELECT rowid, metadata_json FROM paper_assets '
        'WHERE paper_id = ? AND blob_id = ? AND role = ?',
        (paper_id, blob_id, SUPPLEMENT_ROLE),
    ).fetchall()
    for row in rows:
        details = json.loads(row['metadata_json'])
        details[f'{stage}_extraction_status'] = status
        details[f'{stage}_extraction_error'] = error
        if members is not None:
            details[f'{stage}_extraction_members'] = members
        conn.execute('UPDATE paper_assets SET metadata_json = ? WHERE rowid = ?',
                     (json.dumps(details, sort_keys=True), row['rowid']))
    conn.commit()
    return bool(rows)
