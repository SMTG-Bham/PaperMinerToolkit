"""Unit tests for paperminer.corpus.documents.

This module tests text extraction from PDF/TXT inputs, PDF image extraction,
and page rendering helpers.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import types
from typing import Any

import pytest

import paperminer.corpus.documents as documents


def test_pdf_bytes_delegates_to_stream_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wrap byte content in a file-like object for the common PDF reader."""
    monkeypatch.setattr(documents, 'read_pdf_text', lambda stream: stream.read().decode())
    assert documents.read_pdf_bytes(b'paper') == 'paper'


def test_read_pdf_text_concatenates_page_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """Concatenate extracted text from every PDF page."""

    class FakePage:
        """Provide optional extracted page text."""

        def __init__(self, text: str | None) -> None:
            """Store the extracted text."""
            self.text = text

        def extract_text(self) -> str | None:
            """Return the stored page text."""
            return self.text

    class FakeReader:
        """Provide a fake PDF reader with three pages."""

        def __init__(self, path: str) -> None:
            """Store the path and construct fake pages."""
            self.path = path
            self.pages = [FakePage('first '), FakePage(None), FakePage('third')]

    monkeypatch.setattr(documents, 'PdfReader', FakeReader)

    assert documents.read_pdf_text('paper.pdf') == 'first third'


def test_read_document_text_reads_text_files_and_pdf_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Read TXT and PDF inputs with optional reference trimming."""
    text_path = tmp_path / 'paper.txt'
    text_path.write_text('plain text')
    monkeypatch.setattr(documents, 'read_pdf_text', lambda _: 'Main text\nReferences\nReference one')

    assert documents.read_document_text(str(text_path)) == 'plain text'
    assert documents.read_document_text('paper.pdf') == 'Main text\n'
    assert documents.read_document_text('paper.pdf', trim_references=False) == 'Main text\nReferences\nReference one'


def test_read_document_text_rejects_unsupported_file_types() -> None:
    """Reject unsupported document extensions."""
    with pytest.raises(ValueError, match='Unsupported document type'):
        documents.read_document_text('paper.docx')


def test_load_fitz_returns_module_or_raises_helpful_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Load PyMuPDF lazily or raise a helpful dependency error."""
    fake_fitz = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, 'fitz', fake_fitz)
    assert documents._load_fitz() is fake_fitz

    original_import = __import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        """Reject PyMuPDF imports and delegate all other imports."""
        if name == 'fitz':
            raise ImportError('missing')
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, 'fitz', raising=False)
    monkeypatch.setattr('builtins.__import__', fake_import)
    with pytest.raises(RuntimeError, match='Image analysis requires PyMuPDF'):
        documents._load_fitz()


def test_extract_embedded_pdf_images_saves_unique_images(tmp_path: Path) -> None:
    """Save each unique embedded PDF image once."""

    class FakePage:
        """Provide embedded image references for a fake page."""

        def __init__(self, images: list[tuple[int, ...]]) -> None:
            """Store the embedded image references."""
            self.images = images

        def get_images(self, full: bool = True) -> list[tuple[int, ...]]:
            """Return the stored full image references."""
            assert full is True
            return self.images

    class FakeDoc:
        """Provide indexed pages and extractable images."""

        pages = [FakePage([(1,), (2,)]), FakePage([(1,), (3,)])]

        def __len__(self) -> int:
            """Return the number of fake pages."""
            return len(self.pages)

        def __getitem__(self, index: int) -> FakePage:
            """Return a fake page by index."""
            return self.pages[index]

        def extract_image(self, xref: int) -> dict[str, str | bytes]:
            """Return deterministic bytes for an image reference."""
            return {'ext': 'bin', 'image': f'image-{xref}'.encode()}

    saved = documents._extract_embedded_pdf_images(FakeDoc(), str(tmp_path), 'paper')

    assert [os.path.basename(path) for path in saved] == [
        'paper_page-1_image-1.bin',
        'paper_page-1_image-2.bin',
        'paper_page-2_image-2.bin',
    ]
    assert [open(path, 'rb').read() for path in saved] == [b'image-1', b'image-2', b'image-3']


def test_render_pdf_pages_saves_rendered_pages(tmp_path: Path) -> None:
    """Render each PDF page with the expected zoom matrix."""

    class FakeFitz:
        """Provide a fake PyMuPDF matrix factory."""

        @staticmethod
        def Matrix(x_zoom: float, y_zoom: float) -> tuple[str, float, float]:
            """Return a tuple representing a render matrix."""
            return ('matrix', x_zoom, y_zoom)

    class FakePixmap:
        """Track the path where a rendered pixmap is saved."""

        def __init__(self) -> None:
            """Initialize without a saved path."""
            self.saved_path = None

        def save(self, path: str) -> None:
            """Record the path and write placeholder PNG data."""
            self.saved_path = path
            with open(path, 'w', encoding='utf-8') as f:
                f.write('png')

    class FakePage:
        """Provide a page that records rendering arguments."""

        def __init__(self) -> None:
            """Initialize the page and its fake pixmap."""
            self.matrix = None
            self.pixmap = FakePixmap()

        def get_pixmap(
            self,
            matrix: tuple[str, float, float],
            alpha: bool = False,
        ) -> FakePixmap:
            """Record rendering options and return the fake pixmap."""
            self.matrix = matrix
            assert alpha is False
            return self.pixmap

    class FakeDoc:
        """Provide two indexable fake pages."""

        def __init__(self) -> None:
            """Construct the fake pages."""
            self.pages = [FakePage(), FakePage()]

        def __len__(self) -> int:
            """Return the number of fake pages."""
            return len(self.pages)

        def __getitem__(self, index: int) -> FakePage:
            """Return a fake page by index."""
            return self.pages[index]

    doc = FakeDoc()

    saved = documents._render_pdf_pages(FakeFitz, doc, str(tmp_path), 'paper', dpi=144)

    assert [os.path.basename(path) for path in saved] == ['paper_page-1.png', 'paper_page-2.png']
    assert doc.pages[0].matrix == ('matrix', 2.0, 2.0)
    assert all(os.path.isfile(path) for path in saved)


def test_extract_pdf_images_validates_strategy_and_uses_embedded_or_rendered_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate and dispatch PDF image extraction strategies."""

    class FakeDoc:
        """Provide a context-managed fake PDF document."""

        def __enter__(self) -> FakeDoc:
            """Return the fake document."""
            return self

        def __exit__(self, *_: Any) -> bool:
            """Propagate exceptions raised in the context."""
            return False

    class FakeFitz:
        """Provide a fake PyMuPDF document opener."""

        @staticmethod
        def open(path: str) -> FakeDoc:
            """Return a context-managed fake document."""
            return FakeDoc()

    with pytest.raises(ValueError, match='strategy'):
        documents.extract_pdf_images('paper.pdf', str(tmp_path), strategy='bad')

    monkeypatch.setattr(documents, '_load_fitz', lambda: FakeFitz)
    monkeypatch.setattr(documents, '_extract_embedded_pdf_images', lambda *_: ['embedded.png'])
    monkeypatch.setattr(documents, '_render_pdf_pages', lambda *_args, **_kwargs: ['rendered.png'])

    assert documents.extract_pdf_images('paper.pdf', str(tmp_path), strategy='embedded') == ['embedded.png']

    monkeypatch.setattr(documents, '_extract_embedded_pdf_images', lambda *_: [])
    assert documents.extract_pdf_images('paper.pdf', str(tmp_path), strategy='auto') == ['rendered.png']
    assert documents.extract_pdf_images('paper.pdf', str(tmp_path), strategy='pages') == ['rendered.png']
