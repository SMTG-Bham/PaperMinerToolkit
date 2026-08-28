"""Tests for downloading figure assets from structured document layouts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pytest
import requests

from paperminertoolkit.corpus import database
from paperminertoolkit.corpus.layout import Figure, Graphic
from paperminertoolkit.workflows import figures


PNG = b'\x89PNG\r\n\x1a\nfigure bytes'


class ImageResponse:
    """Return prepared binary content from an injectable HTTP session."""

    def __init__(
        self,
        content: object = PNG,
        *,
        status_code: int = 200,
        headers: Mapping[str, str] | None = None,
        url: str = '',
    ) -> None:
        """Store response bytes, status, headers, and optional redirect URL."""
        self.content = content
        self.status_code = status_code
        self.headers = dict(headers or {'Content-Type': 'image/png'})
        self.url = url
        self.text = ''

    def json(self) -> Any:
        """Reject JSON decoding because image responses are binary."""
        raise ValueError('binary response')

    def raise_for_status(self) -> None:
        """Raise an HTTP error for a prepared unsuccessful status."""
        if self.status_code >= 400:
            raise requests.HTTPError(f'{self.status_code} error', response=self)


class ImageSession:
    """Return binary responses in order and record requests."""

    def __init__(self, responses: Iterable[ImageResponse]) -> None:
        """Store the prepared response iterator."""
        self.responses = iter(responses)
        self.calls: list[dict[str, object]] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, object],
        headers: Mapping[str, str],
        timeout: float,
    ) -> ImageResponse:
        """Record request arguments and return the next response."""
        self.calls.append({
            'url': url,
            'params': dict(params),
            'headers': dict(headers),
            'timeout': timeout,
        })
        response = next(self.responses)
        if not response.url:
            response.url = url
        return response


def _paper(paper_id: str = 'paper:figures') -> dict[str, object]:
    """Return minimal metadata for a paper with a declared licence."""
    return {
        'paper_id': paper_id,
        'doi': '10.1000/figures',
        'title': 'Figure paper',
        'license': 'CC BY 4.0',
    }


def _add_jats(db_path: Path, content: bytes, *, document_format: str = 'jats') -> None:
    """Create a corpus paper and attach one structured document."""
    with database.connect(db_path) as conn:
        database.add_structured_document(
            conn,
            _paper(),
            content,
            document_format=document_format,
            source='pubmed',
            original_filename='article.xml',
            metadata={
                'source_url': 'https://repo.example/articles/article.xml',
                'source_identifier': 'PMC1',
                'license': 'CC BY 4.0',
            },
        )


def test_downloads_redirects_deduplicates_resumes_and_forces(tmp_path: Path) -> None:
    """Store linked images once, follow redirects, resume, and permit refreshes."""
    db_path = tmp_path / 'corpus.db'
    _add_jats(
        db_path,
        b'''<article xmlns:xlink="http://www.w3.org/1999/xlink"><body><sec id="results">
        <fig id="fig-1"><label>Figure 1</label><caption><p>First caption.</p></caption>
          <graphic id="graphic-1" xlink:href="../images/first" mime-type="image/png"/>
        </fig>
        <fig id="fig-2"><label>Figure 2</label><caption><p>Second caption.</p></caption>
          <graphic id="graphic-2" xlink:href="https://cdn.example/second.png"/>
        </fig></sec></body></article>''',
    )
    session = ImageSession([
        ImageResponse(status_code=503),
        ImageResponse(url='https://cdn.example/redirected-first.png'),
        ImageResponse(),
    ])

    summary = figures.download_structured_figures(db_path, 'paper:figures', session=session)

    assert summary == figures.FigureDownloadSummary(2, 0, 0)
    assert len(session.calls) == 3
    assert session.calls[0]['url'] == 'https://repo.example/images/first'
    assert session.calls[0]['params'] == {}
    assert session.calls[0]['headers']['Accept'] == 'image/*'
    with database.connect(db_path) as conn:
        assets = database.get_figure_assets(conn, 'paper:figures', include_content=True)
        blobs = conn.execute(
            'SELECT COUNT(*) FROM blobs WHERE kind = ?', ('image',),
        ).fetchone()[0]
    assert len(assets) == 2
    assert blobs == 1
    assert {asset['metadata']['figure_id'] for asset in assets} == {'fig-1', 'fig-2'}
    first = next(asset for asset in assets if asset['metadata']['figure_id'] == 'fig-1')
    assert first['metadata']['caption'] == 'First caption.'
    assert first['metadata']['graphic_id'] == 'graphic-1'
    assert first['metadata']['section_id'] == 'results'
    assert first['metadata']['requested_url'] == 'https://repo.example/images/first'
    assert first['metadata']['source_url'] == 'https://cdn.example/redirected-first.png'
    assert first['metadata']['license'] == 'CC BY 4.0'
    assert first['content'] == PNG

    resume_session = ImageSession([])
    resumed = figures.download_structured_figures(
        db_path, 'paper:figures', session=resume_session,
    )
    assert resumed == figures.FigureDownloadSummary(0, 2, 0)
    assert resume_session.calls == []

    force_session = ImageSession([ImageResponse(), ImageResponse()])
    refreshed = figures.download_structured_figures(
        db_path, 'paper:figures', force=True, session=force_session,
    )
    assert refreshed == figures.FigureDownloadSummary(2, 0, 0)
    assert len(force_session.calls) == 2


def test_reports_document_url_and_payload_failures_without_stopping(tmp_path: Path) -> None:
    """Continue after malformed layouts, unsafe URLs, empty bodies, and invalid images."""
    db_path = tmp_path / 'failures.db'
    _add_jats(
        db_path,
        b'''<article xmlns:xlink="http://www.w3.org/1999/xlink"><body>
        <fig id="unsafe"><graphic xlink:href="file:///tmp/private.png"/></fig>
        <fig id="html"><graphic xlink:href="images/not-image"/></fig>
        <fig id="empty"><graphic xlink:href="images/empty"/></fig>
        <fig id="large"><graphic xlink:href="images/large"/></fig>
        </body></article>''',
    )
    with database.connect(db_path) as conn:
        database.add_structured_document(
            conn,
            _paper(),
            b'<article>',
            document_format='jats',
            source='broken',
            original_filename='broken.xml',
        )
        database.add_structured_document(
            conn,
            _paper(),
            b'<other/>',
            document_format='custom-xml',
            source='custom',
            original_filename='custom.xml',
        )
    session = ImageSession([
        ImageResponse(b'<html>not an image</html>', headers={'Content-Type': 'text/html'}),
        ImageResponse(b''),
        ImageResponse(PNG, headers={'Content-Type': 'image/png', 'Content-Length': '1000'}),
    ])

    summary = figures.download_structured_figures(
        db_path, 'paper:figures', session=session, max_bytes=100,
    )

    assert summary.downloaded == 0
    assert summary.skipped == 0
    assert summary.failed == 6
    assert any('unsupported structured-document format' in item for item in summary.failures)
    assert any('malformed jats document' in item for item in summary.failures)
    assert any('HTTP or HTTPS' in item for item in summary.failures)
    assert any('not a supported image' in item for item in summary.failures)
    assert any('response is empty' in item for item in summary.failures)
    assert any('exceeds the 100-byte limit' in item for item in summary.failures)
    with database.connect(db_path) as conn:
        assert database.get_figure_assets(conn, 'paper:figures') == []


def test_download_rejects_invalid_arguments_and_handles_no_documents(tmp_path: Path) -> None:
    """Validate run arguments and make an empty existing paper a no-op."""
    db_path = tmp_path / 'empty.db'
    with database.connect(db_path) as conn:
        database.upsert_paper(conn, _paper())
    assert figures.download_structured_figures(
        db_path, 'paper:figures', session=ImageSession([]),
    ) == figures.FigureDownloadSummary(0, 0, 0)
    with pytest.raises(ValueError, match='max_bytes must be positive'):
        figures.download_structured_figures(db_path, 'paper:figures', max_bytes=0)
    with pytest.raises(ValueError, match='paper not found'):
        figures.download_structured_figures(db_path, 'missing')


@pytest.mark.parametrize(
    ('uri', 'base_url', 'message'),
    [
        ('', 'https://example.org/article.xml', 'empty'),
        ('file:///tmp/image.png', 'https://example.org/article.xml', 'HTTP or HTTPS'),
        ('https://user:pass@example.org/image.png', '', 'credentials'),
        ('https://localhost/image.png', '', 'local host'),
        ('https://127.0.0.1/image.png', '', 'public network'),
        ('https:///image.png', '', 'no host'),
    ],
)
def test_safe_image_url_rejects_unsafe_references(
    uri: str,
    base_url: str,
    message: str,
) -> None:
    """Reject URL forms that should never be requested from article markup."""
    with pytest.raises(ValueError, match=message):
        figures._safe_image_url(uri, base_url)


def test_safe_image_url_resolves_relative_and_absolute_references() -> None:
    """Resolve repository paths and remove fragments from absolute URLs."""
    assert figures._safe_image_url(
        '../figures/a.png#panel', 'https://repo.example/articles/paper.xml',
    ) == 'https://repo.example/figures/a.png'
    assert figures._safe_image_url(
        'HTTPS://CDN.EXAMPLE/a.png?download=1#fragment', 'https://repo.example/paper.xml',
    ) == 'https://CDN.EXAMPLE/a.png?download=1'


@pytest.mark.parametrize(
    ('content', 'mime_type'),
    [
        (PNG, 'image/png'),
        (b'\xff\xd8\xffjpeg', 'image/jpeg'),
        (b'GIF89a-data', 'image/gif'),
        (b'II*\x00tiff', 'image/tiff'),
        (b'RIFF1234WEBPdata', 'image/webp'),
        (b'<?xml version="1.0"?><svg></svg>', 'image/svg+xml'),
    ],
)
def test_image_signature_validation_accepts_supported_formats(
    content: bytes,
    mime_type: str,
) -> None:
    """Recognize every supported raster and SVG image signature."""
    assert figures._validated_mime_type(content, (mime_type,)) == mime_type


def test_image_signature_validation_rejects_mismatch_and_active_svg() -> None:
    """Reject contradictory declarations and SVGs containing active content."""
    with pytest.raises(ValueError, match='does not match'):
        figures._validated_mime_type(PNG, ('image/jpeg',))
    with pytest.raises(ValueError, match='active or entity-bearing'):
        figures._validated_mime_type(b'<svg><script/></svg>', ('image/svg+xml',))
    with pytest.raises(ValueError, match='active or entity-bearing'):
        figures._validated_mime_type(b'<svg><!ENTITY x "y"></svg>', ('image/svg+xml',))


def test_response_and_filename_helpers_cover_sparse_metadata() -> None:
    """Handle invalid lengths, non-byte bodies, and missing source filenames."""
    response = ImageResponse(PNG, headers={'Content-Type': 'image/png', 'Content-Length': 'nope'})
    assert figures._response_content(response, 100) == PNG
    with pytest.raises(ValueError, match='empty'):
        figures._response_content(ImageResponse('not bytes'), 100)
    with pytest.raises(ValueError, match='exceeds'):
        figures._response_content(ImageResponse(PNG), 2)
    name = figures._asset_filename(
        Figure('figure 1'), Graphic(), 'https://example.org', 'image/jpeg',
    )
    assert name.startswith('figure-1-figure-1-')
    assert name.endswith('.jpg')


def test_request_headers_use_optional_elsevier_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Authenticate Elsevier image requests and fall back cleanly without a key."""
    monkeypatch.setattr(figures.elsevier, 'configured_api_key', lambda: 'elsevier-key')
    assert figures._request_headers('Elsevier')['X-ELS-APIKey'] == 'elsevier-key'

    def missing_key() -> str:
        """Simulate absent optional Elsevier credentials."""
        raise ValueError('missing')

    monkeypatch.setattr(figures.elsevier, 'configured_api_key', missing_key)
    headers = figures._request_headers('elsevier')
    assert headers['Accept'] == 'image/*'
    assert 'X-ELS-APIKey' not in headers


@pytest.mark.parametrize(
    ('document_format', 'content'),
    [
        ('elsevier-xml', '<article><body><figure id="f1"/></body></article>'),
        ('tei', '<TEI><text><body><figure xml:id="f1"/></body></text></TEI>'),
    ],
)
def test_layout_from_non_jats_structured_assets(
    document_format: str,
    content: str,
) -> None:
    """Dispatch publisher XML and PDF-derived TEI to their layout adapters."""
    layout = figures._layout_from_asset(
        {
            'source': 'elsevier' if document_format == 'elsevier-xml' else 'openalex',
            'content': content,
            'metadata': {
                'document_format': document_format,
                'source_identifier': 'source-1',
            },
        },
        'paper:1',
    )
    assert layout.document_id == 'paper:1'
    assert layout.provenance.document_format == document_format
