import os
from pathlib import Path

from PyPDF2 import PdfReader

from paperscraper.pipeline import existing_path


def read_pdf_text(pdf_path: str):
    reader = PdfReader(pdf_path)
    text = ''
    for page in reader.pages:
        page_text = page.extract_text() or ''
        text += page_text
    return text


def read_document_text(path: str, trim_references: bool = True):
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


def extract_pdf_images(pdf_path: str, output_dir: str, prefix: str | None = None):
    try:
        import fitz
    except ImportError as e:
        raise RuntimeError('Image analysis requires PyMuPDF. Install the package with: pip install pymupdf') from e

    os.makedirs(output_dir, exist_ok=True)
    prefix = prefix or Path(pdf_path).stem
    saved = []
    with fitz.open(pdf_path) as doc:
        for page_index in range(len(doc)):
            page = doc[page_index]
            for image_index, image in enumerate(page.get_images(full=True), start=1):
                xref = image[0]
                extracted = doc.extract_image(xref)
                ext = extracted.get('ext', 'png')
                image_bytes = extracted['image']
                filename = f'{prefix}_page-{page_index + 1}_image-{image_index}.{ext}'
                filepath = os.path.join(output_dir, filename)
                with open(filepath, 'wb') as out_file:
                    out_file.write(image_bytes)
                saved.append(filepath)
    return saved


def text_file_for_row(papers_dir, files, row):
    return existing_path(row.get('text_path')) or file_for_row_by_identifier(papers_dir, files, row, '.txt')


def pdf_file_for_row(papers_dir, files, row):
    return existing_path(row.get('pdf_path')) or file_for_row_by_identifier(papers_dir, files, row, '.pdf')


def file_for_row_by_identifier(papers_dir, files, row, extension):
    scopus_id = row['dc:identifier'].split(':')[-1]
    filenames = [file for file in files if scopus_id in file and file.lower().endswith(extension)]
    if filenames:
        return os.path.join(papers_dir, filenames[0])
    return None
