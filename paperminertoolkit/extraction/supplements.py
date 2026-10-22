"""Read supplementary files without executing documents or extracting archives.

Archive and Office documents are inspected in memory with bounded expansion.
The original corpus asset remains the authoritative source for every part.
"""

from __future__ import annotations

import io
import math
import mimetypes
import re
import stat
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any
from xml.etree import ElementTree

from paperminertoolkit.corpus.documents import extract_pdf_images, read_pdf_bytes

MAX_ARCHIVE_ENTRIES = 256
MAX_MEMBER_BYTES = 64 * 1024 * 1024
MAX_EXPANDED_BYTES = 128 * 1024 * 1024
MAX_COMPRESSION_RATIO = 1000
MAX_PDF_PAGES = 256
MAX_RENDERED_PIXELS = 250_000_000
_TEXT_SUFFIXES = {'.txt', '.csv', '.tsv', '.json', '.xml', '.html', '.htm', '.md', '.rst'}
_IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.tif', '.tiff', '.bmp'}


@dataclass(frozen=True, slots=True)
class SupplementPart:
    """One supplementary file or an identified member of a ZIP archive."""

    name: str
    content: bytes
    mime_type: str
    kind: str
    member: str = ''

    @property
    def supports_text(self) -> bool:
        """Return whether the part has a supported text reader."""
        return self.kind in {'text', 'pdf', 'docx', 'xlsx'}

    @property
    def supports_images(self) -> bool:
        """Return whether the part can supply vision model inputs."""
        return self.kind in {'pdf', 'image'}


def _archive_contents(content: bytes) -> dict[str, bytes]:
    """Read a bounded archive, rejecting ambiguous or unsafe member names."""
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        entries = archive.infolist()
        if len(entries) > MAX_ARCHIVE_ENTRIES:
            raise ValueError(f'Supplement archive exceeds {MAX_ARCHIVE_ENTRIES} entries')
        expanded = 0
        names: set[str] = set()
        for entry in entries:
            name = entry.filename
            path = PurePosixPath(name)
            if (path.is_absolute() or '..' in path.parts or '\\' in name
                    or re.match(r'^[A-Za-z]:', name) or '\x00' in name
                    or name in names or stat.S_ISLNK(entry.external_attr >> 16)):
                raise ValueError(f'Unsafe or duplicate supplementary archive member: {name!r}')
            names.add(name)
            if entry.flag_bits & 1:
                raise ValueError('Encrypted supplementary archives are unsupported')
            expanded += entry.file_size
            if entry.file_size > MAX_MEMBER_BYTES or expanded > MAX_EXPANDED_BYTES:
                raise ValueError('Supplement archive exceeds the decompressed size limit')
            if entry.file_size > max(entry.compress_size, 1) * MAX_COMPRESSION_RATIO:
                raise ValueError('Supplement archive exceeds the compression ratio limit')
        output = {}
        for entry in entries:
            if entry.is_dir():
                continue
            with archive.open(entry) as member:
                data = member.read(MAX_MEMBER_BYTES + 1)
            if len(data) > MAX_MEMBER_BYTES:
                raise ValueError('Supplement archive member exceeds the size limit')
            output[entry.filename] = data
        return output


def _part(name: str, content: bytes, mime_type: str, member: str = '') -> SupplementPart:
    """Identify a file by its content signature, media type, and extension."""
    suffix = Path(name).suffix.lower()
    mime_type = mime_type.split(';')[0].strip().lower()
    if content.lstrip().startswith(b'%PDF-') or suffix == '.pdf' or mime_type == 'application/pdf':
        kind = 'pdf'
    elif suffix == '.docx' or mime_type.endswith('wordprocessingml.document'):
        kind = 'docx'
    elif suffix == '.xlsx' or mime_type.endswith('spreadsheetml.sheet'):
        kind = 'xlsx'
    elif suffix in _IMAGE_SUFFIXES or mime_type.startswith('image/'):
        kind = 'image'
    elif suffix in _TEXT_SUFFIXES or mime_type.startswith('text/') or mime_type in {
        'application/json', 'application/xml', 'application/xhtml+xml',
    }:
        kind = 'text'
    elif suffix == '.zip' or mime_type in {'application/zip', 'application/x-zip-compressed'} or content.startswith(b'PK\x03\x04'):
        kind = 'zip'
    else:
        kind = 'unsupported'
    return SupplementPart(name, content, mime_type, kind, member)


def supplement_parts(asset: Mapping[str, Any]) -> list[SupplementPart]:
    """Return supported and unsupported parts while preserving their identity.

    ZIP members retain their complete member names for provenance. Nested
    archives remain unsupported files rather than expanding recursively.
    """
    original = str(asset.get('original_filename') or asset['blob_id'])
    part = _part(original, asset['content'], str(asset.get('mime_type') or ''))
    if part.kind != 'zip':
        return [part]
    entries = _archive_contents(part.content)
    if not entries:
        raise ValueError('Supplement archive contains no files')
    return [_part(name, content, mimetypes.guess_type(name)[0] or '', member=name)
            for name, content in entries.items()]


def _decode(content: bytes) -> str:
    """Decode common text encodings without treating binary files as text."""
    if content.startswith((b'\xff\xfe', b'\xfe\xff')):
        return content.decode('utf-16')
    if b'\x00' in content:
        raise ValueError('Supplement contains binary data rather than supported text')
    try:
        return content.decode('utf-8-sig')
    except UnicodeDecodeError:
        return content.decode('cp1252')


def _xml(content: bytes) -> ElementTree.Element:
    """Parse XML with document type definitions and entities disabled."""
    text = _decode(content)
    if re.search(r'<!\s*(?:DOCTYPE|ENTITY)\b', text, flags=re.IGNORECASE):
        raise ValueError('Document type definitions and entities are unsupported in supplementary XML')
    return ElementTree.fromstring(text)


class _HTMLText(HTMLParser):
    """Collect readable HTML text while excluding script and style content."""

    def __init__(self) -> None:
        """Initialize the collected text and exclusion depth."""
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.ignored = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Start ignored blocks or separate readable elements."""
        if tag in {'script', 'style'}:
            self.ignored += 1
        if tag in {'p', 'div', 'tr', 'br', 'h1', 'h2', 'h3', 'li'}:
            self.parts.append('\n')
        if tag in {'td', 'th'}:
            self.parts.append('\t')

    def handle_endtag(self, tag: str) -> None:
        """End an ignored block."""
        if tag in {'script', 'style'} and self.ignored:
            self.ignored -= 1

    def handle_data(self, data: str) -> None:
        """Collect visible character data."""
        if not self.ignored:
            self.parts.append(data)


def _office_text(part: SupplementPart) -> str:
    """Read DOCX paragraphs or XLSX cached cells without executing content."""
    files = _archive_contents(part.content)
    if part.kind == 'docx':
        root = _xml(files['word/document.xml'])
        namespace = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        return '\n'.join(''.join(paragraph.itertext()) for paragraph in root.findall('.//w:p', namespace))
    ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    shared = []
    if 'xl/sharedStrings.xml' in files:
        shared = [''.join(node.itertext()) for node in _xml(files['xl/sharedStrings.xml']).findall('s:si', ns)]
    sheet_names: dict[str, str] = {}
    if 'xl/workbook.xml' in files and 'xl/_rels/workbook.xml.rels' in files:
        relationships = {node.get('Id'): node.get('Target', '') for node in _xml(files['xl/_rels/workbook.xml.rels'])}
        for sheet in _xml(files['xl/workbook.xml']).findall('s:sheets/s:sheet', ns):
            target = relationships.get(sheet.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'), '')
            path = target.lstrip('/') if target.startswith('/') else str(PurePosixPath('xl') / target)
            sheet_names[path] = sheet.get('name', path)
    output = []
    for name, content in sorted(files.items()):
        if not re.fullmatch(r'xl/worksheets/[^/]+\.xml', name):
            continue
        output.append(f'Sheet: {sheet_names.get(name, name)}')
        for row in _xml(content).findall('s:sheetData/s:row', ns):
            cells = []
            for cell in row.findall('s:c', ns):
                value = cell.findtext('s:v', default='', namespaces=ns)
                if cell.get('t') == 's' and value:
                    value = shared[int(value)]
                elif cell.get('t') == 'inlineStr':
                    value = ''.join(cell.find('s:is', ns).itertext())
                elif not value and cell.find('s:f', ns) is not None:
                    value = f'={cell.findtext("s:f", namespaces=ns)} [no cached value]'
                cells.append(f'{cell.get("r", "")}: {value}')
            output.append('\t'.join(cells))
    return '\n'.join(output)


def supplement_text(part: SupplementPart) -> str:
    """Read supported SI text, retaining references and raw tabular values."""
    if part.kind == 'pdf':
        return read_pdf_bytes(part.content)
    if part.kind in {'docx', 'xlsx'}:
        return _office_text(part)
    if part.kind != 'text':
        raise ValueError(f'Unsupported supplementary text format: {part.name}')
    suffix = Path(part.name).suffix.lower()
    if suffix == '.xml' or part.mime_type == 'application/xml':
        return '\n'.join(text.strip() for text in _xml(part.content).itertext() if text.strip())
    text = _decode(part.content)
    if suffix in {'.html', '.htm'} or part.mime_type in {'text/html', 'application/xhtml+xml'}:
        parser = _HTMLText()
        parser.feed(text)
        return ''.join(parser.parts)
    return text


def supplement_images(part: SupplementPart, directory: str, *, dpi: int = 200) -> list[str]:
    """Render SI PDF pages or normalize raster frames to PNG files.

    Full PDF pages preserve supplementary tables and figures together with
    their labels. Directory names are assigned by the caller, never by archive
    members. No source archive path is used to write a file.
    """
    from PIL import Image

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    if part.kind == 'pdf':
        import fitz

        if dpi < 1:
            raise ValueError('Supplement PDF image DPI must be positive')
        with fitz.open(stream=part.content, filetype='pdf') as document:
            if len(document) > MAX_PDF_PAGES:
                raise ValueError(f'Supplement PDF exceeds the {MAX_PDF_PAGES} page rendering limit')
            pixels = sum(math.ceil(page.rect.width * dpi / 72)
                         * math.ceil(page.rect.height * dpi / 72) for page in document)
            if pixels > MAX_RENDERED_PIXELS:
                raise ValueError('Supplement PDF exceeds the total rendered pixel limit; use a lower DPI')
        pdf_path = destination / 'source.pdf'
        pdf_path.write_bytes(part.content)
        try:
            return extract_pdf_images(str(pdf_path), str(destination), prefix='si', strategy='pages', dpi=dpi)
        finally:
            pdf_path.unlink(missing_ok=True)
    if part.kind != 'image':
        raise ValueError(f'Unsupported supplementary image format: {part.name}')
    output = []
    with Image.open(io.BytesIO(part.content)) as document:
        frames = getattr(document, 'n_frames', 1)
        if frames > MAX_ARCHIVE_ENTRIES:
            raise ValueError('Supplement image exceeds the supported frame count')
        pixels = 0
        for index in range(frames):
            document.seek(index)
            pixels += document.width * document.height
            if pixels > MAX_RENDERED_PIXELS:
                raise ValueError('Supplement image exceeds the total rendered pixel limit')
            path = destination / f'si-{index + 1}.png'
            document.convert('RGB').save(path)
            output.append(str(path))
    return output
