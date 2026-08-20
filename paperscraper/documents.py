"""Read text from PDFs and extract images/pages for vision analysis.

This module centralizes document IO helpers used by scraping: text extraction
from PDFs, text/PDF path lookup for paper rows, and image extraction/rendering
for figure or page-level model analysis.
"""

import io
import os
import re
from pypdf import PdfReader
from pathlib import Path


def read_pdf_text(pdf_path):
    """Extract concatenated text from every page in a PDF.

    Parameters
    ----------
    pdf_path : str or file-like object
        PDF path or binary stream accepted by :class:`pypdf.PdfReader`.

    Returns
    -------
    str
        Page text in document order. Pages without extractable text contribute
        an empty string.
    """
    reader = PdfReader(pdf_path)
    text = ''
    for page in reader.pages:
        page_text = page.extract_text() or ''
        text += page_text
    return text


def read_pdf_bytes(content: bytes):
    """Extract text directly from in-memory PDF bytes.

    Parameters
    ----------
    content : bytes
        Complete PDF file contents.

    Returns
    -------
    str
        Concatenated text from the PDF pages.
    """
    return read_pdf_text(io.BytesIO(content))


def trim_reference_section(text: str):
    """Remove a trailing reference section identified by a standalone heading.

    Parameters
    ----------
    text : str
        Document text to trim.

    Returns
    -------
    str
        Text before the final references heading, or the original text when no
        heading is present.
    """
    headings = list(re.finditer(
        r'(?im)^\s*(?:\d+(?:\.\d+)*[.)]?\s+)?(?:references|bibliography|literature cited)\s*$',
        text,
    ))
    if headings:
        return text[:headings[-1].start()]
    return text


def read_document_text(path: str, trim_references: bool = True):
    """Read text from a TXT or PDF document.

    Parameters
    ----------
    path : str
        Path to a ``.txt`` or ``.pdf`` document.
    trim_references : bool, default=True
        Whether to remove the trailing reference section from PDFs.

    Returns
    -------
    str
        Extracted document text.

    Raises
    ------
    ValueError
        If ``path`` does not have a supported extension.
    """
    if path.lower().endswith('.txt'):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    if path.lower().endswith('.pdf'):
        text = read_pdf_text(path)
        if trim_references:
            text = trim_reference_section(text)
        return text
    raise ValueError(f'Unsupported document type: {path}')


def _load_fitz():
    """Import PyMuPDF lazily.

    Returns
    -------
    module
        Imported :mod:`fitz` module.

    Raises
    ------
    RuntimeError
        If PyMuPDF is not installed.
    """
    try:
        import fitz
    except ImportError as e:
        raise RuntimeError('Image analysis requires PyMuPDF. Install the package with: pip install pymupdf') from e
    return fitz


def _extract_embedded_pdf_images(doc, output_dir: str, prefix: str):
    """Save unique embedded images from an opened PyMuPDF document.

    Parameters
    ----------
    doc : fitz.Document
        Open PyMuPDF document.
    output_dir : str
        Directory where extracted images are written.
    prefix : str
        Filename prefix for saved images.

    Returns
    -------
    list of str
        Paths to saved images in page and image order.
    """
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
    """Render each PDF page to a PNG image file.

    Parameters
    ----------
    fitz : module
        Imported PyMuPDF module.
    doc : fitz.Document
        Open PyMuPDF document.
    output_dir : str
        Directory where rendered pages are written.
    prefix : str
        Filename prefix for rendered pages.
    dpi : int, default=200
        Render resolution in dots per inch.

    Returns
    -------
    list of str
        Paths to rendered page images in document order.
    """
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
    """Extract images from a PDF for vision scraping.

    Parameters
    ----------
    pdf_path : str
        Path to the source PDF.
    output_dir : str
        Directory where image files are written.
    prefix : str, optional
        Filename prefix. The PDF stem is used when omitted.
    strategy : {'auto', 'embedded', 'pages'}, default='auto'
        Extraction strategy. ``auto`` uses embedded images when present and
        otherwise renders complete pages.
    dpi : int, default=200
        Resolution used when rendering complete pages.

    Returns
    -------
    list of str
        Paths to extracted or rendered image files.

    Raises
    ------
    ValueError
        If ``strategy`` is not supported.
    RuntimeError
        If PyMuPDF is unavailable.
    """
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
