"""Unit tests for paperscraper.documents.

This module tests text extraction from PDF/TXT inputs, PDF image extraction and
page rendering helpers, and lookup of text/PDF files associated with paper rows.
"""

import importlib
import os
import sys
import types

import pytest

documents = importlib.import_module('paperscraper.documents')


def test_read_pdf_text_concatenates_page_text(monkeypatch):
    """
    Test PDF text extraction across all pages.

    This function performs the following steps:
    1. Replaces `PdfReader` with a fake reader containing three fake pages.
    2. Calls `read_pdf_text`.
    3. Compares the concatenated text to the expected page text.

    Asserts:
        - Text from every page is concatenated.
        - Pages with no extracted text contribute an empty string.
    """

    class FakePage:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

    class FakeReader:
        def __init__(self, path):
            self.path = path
            self.pages = [FakePage('first '), FakePage(None), FakePage('third')]

    monkeypatch.setattr(documents, 'PdfReader', FakeReader)

    assert documents.read_pdf_text('paper.pdf') == 'first third'


def test_read_document_text_reads_text_files_and_pdf_files(tmp_path, monkeypatch):
    """
    Test document text reading for TXT and PDF inputs.

    This function performs the following steps:
    1. Writes a temporary TXT file and reads it.
    2. Replaces PDF text extraction with deterministic text containing references.
    3. Reads a PDF path with reference trimming enabled and disabled.

    Asserts:
        - TXT files are read directly.
        - PDF references are trimmed when requested.
        - PDF references are preserved when trimming is disabled.
    """
    text_path = tmp_path / 'paper.txt'
    text_path.write_text('plain text')
    monkeypatch.setattr(documents, 'read_pdf_text', lambda _: 'Main text\nReferences\nReference one')

    assert documents.read_document_text(str(text_path)) == 'plain text'
    assert documents.read_document_text('paper.pdf') == 'Main text\n'
    assert documents.read_document_text('paper.pdf', trim_references=False) == 'Main text\nReferences\nReference one'


def test_read_document_text_rejects_unsupported_file_types():
    """
    Test document text reading for unsupported file types.

    This function performs the following steps:
    1. Calls `read_document_text` with an unsupported extension.
    2. Captures the expected exception.

    Asserts:
        - Unsupported document extensions raise `ValueError`.
    """
    with pytest.raises(ValueError, match='Unsupported document type'):
        documents.read_document_text('paper.docx')


def test_load_fitz_returns_module_or_raises_helpful_error(monkeypatch):
    """
    Test lazy loading of PyMuPDF.

    This function performs the following steps:
    1. Inserts a fake `fitz` module into `sys.modules`.
    2. Calls `_load_fitz` and checks the returned module.
    3. Forces imports of `fitz` to raise `ImportError` and captures the expected error.

    Asserts:
        - Available `fitz` modules are returned.
        - Missing `fitz` raises a helpful `RuntimeError`.
    """
    fake_fitz = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, 'fitz', fake_fitz)
    assert documents._load_fitz() is fake_fitz

    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == 'fitz':
            raise ImportError('missing')
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, 'fitz', raising=False)
    monkeypatch.setattr('builtins.__import__', fake_import)
    with pytest.raises(RuntimeError, match='Image analysis requires PyMuPDF'):
        documents._load_fitz()


def test_extract_embedded_pdf_images_saves_unique_images(tmp_path):
    """
    Test embedded image extraction from a fake PDF document.

    This function performs the following steps:
    1. Creates a fake document with duplicate and unique image references.
    2. Calls `_extract_embedded_pdf_images`.
    3. Reads the saved image files.

    Asserts:
        - Each unique image reference is saved once.
        - Duplicate image references are skipped.
        - Saved files contain the extracted image bytes.
    """

    class FakePage:
        def __init__(self, images):
            self.images = images

        def get_images(self, full=True):
            assert full is True
            return self.images

    class FakeDoc:
        pages = [FakePage([(1,), (2,)]), FakePage([(1,), (3,)])]

        def __len__(self):
            return len(self.pages)

        def __getitem__(self, index):
            return self.pages[index]

        def extract_image(self, xref):
            return {'ext': 'bin', 'image': f'image-{xref}'.encode()}

    saved = documents._extract_embedded_pdf_images(FakeDoc(), str(tmp_path), 'paper')

    assert [os.path.basename(path) for path in saved] == [
        'paper_page-1_image-1.bin',
        'paper_page-1_image-2.bin',
        'paper_page-2_image-2.bin',
    ]
    assert [open(path, 'rb').read() for path in saved] == [b'image-1', b'image-2', b'image-3']


def test_render_pdf_pages_saves_rendered_pages(tmp_path):
    """
    Test rendering PDF pages to image files.

    This function performs the following steps:
    1. Creates fake fitz, document, page, and pixmap objects.
    2. Calls `_render_pdf_pages`.
    3. Checks the saved file paths and render matrix.

    Asserts:
        - One PNG path is returned for each page.
        - Pages receive the expected zoom matrix.
        - Pixmaps save their output files.
    """

    class FakeFitz:
        @staticmethod
        def Matrix(x_zoom, y_zoom):
            return ('matrix', x_zoom, y_zoom)

    class FakePixmap:
        def __init__(self):
            self.saved_path = None

        def save(self, path):
            self.saved_path = path
            with open(path, 'w', encoding='utf-8') as f:
                f.write('png')

    class FakePage:
        def __init__(self):
            self.matrix = None
            self.pixmap = FakePixmap()

        def get_pixmap(self, matrix, alpha=False):
            self.matrix = matrix
            assert alpha is False
            return self.pixmap

    class FakeDoc:
        def __init__(self):
            self.pages = [FakePage(), FakePage()]

        def __len__(self):
            return len(self.pages)

        def __getitem__(self, index):
            return self.pages[index]

    doc = FakeDoc()

    saved = documents._render_pdf_pages(FakeFitz, doc, str(tmp_path), 'paper', dpi=144)

    assert [os.path.basename(path) for path in saved] == ['paper_page-1.png', 'paper_page-2.png']
    assert doc.pages[0].matrix == ('matrix', 2.0, 2.0)
    assert all(os.path.isfile(path) for path in saved)


def test_extract_pdf_images_validates_strategy_and_uses_embedded_or_rendered_paths(tmp_path, monkeypatch):
    """
    Test PDF image extraction strategy selection.

    This function performs the following steps:
    1. Calls `extract_pdf_images` with an invalid strategy.
    2. Replaces PyMuPDF loading and image extraction helpers with local fakes.
    3. Calls `extract_pdf_images` for embedded, auto fallback, and pages strategies.

    Asserts:
        - Invalid strategies raise `ValueError`.
        - Embedded images are returned when available.
        - Auto strategy falls back to rendered pages when no embedded images are found.
        - Pages strategy renders pages directly.
    """

    class FakeDoc:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    class FakeFitz:
        @staticmethod
        def open(path):
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


def test_file_lookup_helpers_prefer_existing_paths_and_match_identifiers(tmp_path):
    """
    Test paper-row file lookup helpers.

    This function performs the following steps:
    1. Creates existing text and PDF files.
    2. Looks up files from explicit row paths.
    3. Looks up files by paper identifier and extension.

    Asserts:
        - Existing explicit paths are preferred.
        - Text files are found by paper identifier.
        - PDF files are found by paper identifier.
        - Missing identifiers return None.
    """
    papers_dir = tmp_path / 'papers'
    papers_dir.mkdir()
    explicit_text = papers_dir / 'explicit.txt'
    explicit_pdf = papers_dir / 'explicit.pdf'
    explicit_text.write_text('text')
    explicit_pdf.write_bytes(b'pdf')
    files = ['abc123_full.txt', 'abc123_full.PDF', 'other.txt']
    row = {'paper_id': 'scopus:abc123', 'text_path': str(explicit_text), 'pdf_path': str(explicit_pdf)}

    assert documents.text_file_for_row(str(papers_dir), files, row) == str(explicit_text)
    assert documents.pdf_file_for_row(str(papers_dir), files, row) == str(explicit_pdf)

    row_without_paths = {'paper_id': 'scopus:abc123', 'text_path': '', 'pdf_path': ''}
    assert documents.text_file_for_row(str(papers_dir), files, row_without_paths) == os.path.join(str(papers_dir), 'abc123_full.txt')
    assert documents.pdf_file_for_row(str(papers_dir), files, row_without_paths) == os.path.join(str(papers_dir), 'abc123_full.PDF')
    assert documents.file_for_row_by_identifier(str(papers_dir), files, {'paper_id': 'scopus:missing'}, '.txt') is None
