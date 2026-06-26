"""Read text from PDFs and extract images/pages for vision analysis.

This module centralizes document IO helpers used by scraping: text extraction
from PDFs, text/PDF path lookup for paper rows, and image extraction/rendering
for figure or page-level model analysis.
"""

import os
from PyPDF2 import PdfReader
from pathlib import Path

from paperscraper.pipeline import existing_path


def read_pdf_text(pdf_path: str):
    """Extract concatenated text from every page in a PDF file."""
    reader = PdfReader(pdf_path)
    text = ''
    for page in reader.pages:
        page_text = page.extract_text() or ''
        text += page_text
    return text


def read_document_text(path: str, trim_references: bool = True):
    """Read text from a TXT or PDF document, optionally trimming references."""
    if path.lower().endswith('.txt'):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    if path.lower().endswith('.pdf'):
        text = read_pdf_text(path)
        if trim_references:
            index = text.lower().rfind('references')
            if index != -1:
                text = text[:index]
        return text
    raise ValueError(f'Unsupported document type: {path}')


def _load_fitz():
    """Import PyMuPDF lazily and raise a helpful error if it is missing."""
    try:
        import fitz
    except ImportError as e:
        raise RuntimeError('Image analysis requires PyMuPDF. Install the package with: pip install pymupdf') from e
    return fitz


def _extract_embedded_pdf_images(doc, output_dir: str, prefix: str):
    """Save unique embedded images from an opened PyMuPDF document."""
    saved = []
    seen_xrefs = set()
    for page_index in range(len(doc)):
        page = doc[page_index]
        for image_index, image in enumerate(page.get_images(full=True), start=1):
            xref = image[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            extracted = doc.extract_image(xref)
            ext = extracted.get('ext', 'png')
            image_bytes = extracted['image']
            filename = f'{prefix}_page-{page_index + 1}_image-{image_index}.{ext}'
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'wb') as out_file:
                out_file.write(image_bytes)
            saved.append(filepath)
    return saved


def _render_pdf_pages(fitz, doc, output_dir: str, prefix: str, dpi: int = 200):
    """Render each PDF page to a PNG image file."""
    saved = []
    zoom = dpi / 72
    render_matrix = fitz.Matrix(zoom, zoom)
    for page_index in range(len(doc)):
        page = doc[page_index]
        pixmap = page.get_pixmap(matrix=render_matrix, alpha=False)
        filepath = os.path.join(output_dir, f'{prefix}_page-{page_index + 1}.png')
        pixmap.save(filepath)
        saved.append(filepath)
    return saved


def extract_pdf_images(pdf_path: str,
                       output_dir: str,
                       prefix: str | None = None,
                       strategy: str = 'auto',
                       dpi: int = 200):
    """Extract embedded images or rendered pages from a PDF for vision scraping."""
    if strategy not in {'auto', 'embedded', 'pages'}:
        raise ValueError('Image extraction strategy must be one of: auto, embedded, pages')

    fitz = _load_fitz()
    os.makedirs(output_dir, exist_ok=True)
    prefix = prefix or Path(pdf_path).stem
    with fitz.open(pdf_path) as doc:
        if strategy in {'auto', 'embedded'}:
            saved = _extract_embedded_pdf_images(doc, output_dir, prefix)
            if saved or strategy == 'embedded':
                return saved
        return _render_pdf_pages(fitz, doc, output_dir, prefix, dpi=dpi)


def text_file_for_row(papers_dir, files, row):
    """Find the text file associated with a paper row."""
    return existing_path(row.get('text_path')) or file_for_row_by_identifier(papers_dir, files, row, '.txt')


def pdf_file_for_row(papers_dir, files, row):
    """Find the PDF file associated with a paper row."""
    return existing_path(row.get('pdf_path')) or file_for_row_by_identifier(papers_dir, files, row, '.pdf')


def file_for_row_by_identifier(papers_dir, files, row, extension):
    """Find a file whose name contains the paper row identifier and extension."""
    paper_id = row['paper_id'].split(':')[-1]
    filenames = [file for file in files if paper_id in file and file.lower().endswith(extension)]
    if filenames:
        return os.path.join(papers_dir, filenames[0])
    return None
