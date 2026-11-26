"""Exercise bounded supplementary readers and independent extraction stages."""

from __future__ import annotations

import io
import sys
import zipfile
from collections.abc import Mapping
from contextlib import closing
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from PIL import Image

from paperminertoolkit.corpus import database as corpus
from paperminertoolkit.corpus.supplements import add_supplement, get_supplements
from paperminertoolkit.extraction import scrape
from paperminertoolkit.extraction import supplements as readers
from tests.extraction.test_scrape import FakeModelConfig, FakeTqdm, sample_recipe


def archive_bytes(files: Mapping[str, bytes]) -> bytes:
    """Create an in-memory archive for an offline reader test."""
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return stream.getvalue()


def asset(content: bytes, name: str, mime_type: str = '') -> dict[str, Any]:
    """Return a minimal corpus asset payload."""
    return {'blob_id': 'asset-123', 'original_filename': name, 'content': content, 'mime_type': mime_type}


def test_archive_members_preserve_names_and_leave_nested_archives_unsupported() -> None:
    """Parse multiple text types without recursively expanding inner archives."""
    payload = archive_bytes({
        'tables/data.csv': b'Name,Value\nLLZO,1.2',
        'values.json': b'{"temperature": 300}',
        'notes.html': b'<p>Supporting results</p><script>hidden</script><table><tr><td>2.5</td></tr></table>',
        'nested.zip': archive_bytes({'data.txt': b'inner'}),
    })
    parts = readers.supplement_parts(asset(payload, 'SI.zip'))
    assert [part.member for part in parts] == ['tables/data.csv', 'values.json', 'notes.html', 'nested.zip']
    assert readers.supplement_text(parts[0]) == 'Name,Value\nLLZO,1.2'
    assert readers.supplement_text(parts[1]) == '{"temperature": 300}'
    assert 'Supporting results' in readers.supplement_text(parts[2])
    assert 'hidden' not in readers.supplement_text(parts[2])
    assert parts[3].kind == 'zip'
    assert not parts[3].supports_text


@pytest.mark.parametrize('name', [
    '../escaped.txt', '/absolute.txt', 'C:/drive.txt',
    # On Windows, zipfile turns a backslash in a member name into "/" both when writing and
    # when reading, so the member is an ordinary "dir/file.txt" there and nothing is ambiguous.
    pytest.param('dir\\file.txt', marks=pytest.mark.skipif(
        sys.platform == 'win32', reason='zipfile normalises backslashes to "/" on Windows')),
])
def test_archive_rejects_unsafe_member_names(name: str) -> None:
    """Reject path traversal and ambiguous archive paths before reading files."""
    with pytest.raises(ValueError, match='Unsafe'):
        readers.supplement_parts(asset(archive_bytes({name: b'x'}), 'SI.zip'))


def test_archive_enforces_entry_expansion_and_ratio_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail closed on oversized archives instead of truncating their contents."""
    payload = archive_bytes({'a.txt': b'a', 'b.txt': b'b'})
    monkeypatch.setattr(readers, 'MAX_ARCHIVE_ENTRIES', 1)
    with pytest.raises(ValueError, match='entries'):
        readers.supplement_parts(asset(payload, 'SI.zip'))
    monkeypatch.setattr(readers, 'MAX_ARCHIVE_ENTRIES', 10)
    monkeypatch.setattr(readers, 'MAX_EXPANDED_BYTES', 1)
    with pytest.raises(ValueError, match='size limit'):
        readers.supplement_parts(asset(payload, 'SI.zip'))
    monkeypatch.setattr(readers, 'MAX_EXPANDED_BYTES', 1_000_000)
    monkeypatch.setattr(readers, 'MAX_COMPRESSION_RATIO', 2)
    with pytest.raises(ValueError, match='compression ratio'):
        readers.supplement_parts(asset(archive_bytes({'big.txt': b'x' * 10000}), 'SI.zip'))


def test_xml_and_office_readers_preserve_data_without_external_entities() -> None:
    """Read OOXML paragraphs and cached spreadsheet cells with stdlib readers."""
    docx = archive_bytes({'word/document.xml': (
        b'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        b'<w:body><w:p><w:r><w:t>Supplementary method</w:t></w:r></w:p>'
        b'<w:tbl><w:tr><w:tc><w:p><w:r><w:t>2.5 S/cm</w:t></w:r></w:p></w:tc></w:tr></w:tbl>'
        b'</w:body></w:document>'
    )})
    assert readers.supplement_text(readers.supplement_parts(asset(docx, 'SI.docx'))[0]) == 'Supplementary method\n2.5 S/cm'
    xlsx = archive_bytes({
        'xl/sharedStrings.xml': b'<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><si><t>LLZO</t></si></sst>',
        'xl/worksheets/sheet1.xml': (
            b'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1">'
            b'<c r="A1" t="s"><v>0</v></c><c r="B1"><f>1+2</f><v>3</v></c>'
            b'<c r="C1"><f>SUM(B1)</f></c></row></sheetData></worksheet>'
        ),
    })
    text = readers.supplement_text(readers.supplement_parts(asset(xlsx, 'SI.xlsx'))[0])
    assert 'A1: LLZO\tB1: 3\tC1: =SUM(B1) [no cached value]' in text
    malicious = b'<!DOCTYPE a [<!ENTITY data SYSTEM "file:///etc/passwd">]><a>&data;</a>'
    with pytest.raises(ValueError, match='entities'):
        readers.supplement_text(readers.supplement_parts(asset(malicious, 'SI.xml'))[0])
    with pytest.raises(ValueError, match='binary'):
        readers.supplement_text(readers.supplement_parts(asset(b'abc\x00def', 'SI.txt'))[0])


def test_xlsx_reader_accepts_empty_inline_string_cells() -> None:
    """Read a worksheet whose writer left an inline-string cell without any text node."""
    xlsx = archive_bytes({'xl/worksheets/sheet1.xml': (
        b'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
        b'<row r="1"><c r="A1" t="inlineStr"><is><t>Soil properties</t></is></c></row>'
        b'<row r="2"><c r="A2" t="inlineStr"></c><c r="B2"><v>6.8</v></c></row>'
        b'</sheetData></worksheet>'
    )})

    text = readers.supplement_text(readers.supplement_parts(asset(xlsx, 'SI.xlsx'))[0])

    assert 'A1: Soil properties' in text
    assert 'A2: \tB2: 6.8' in text


def test_pdf_and_raster_readers_render_full_pages_with_limits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Read real PDF text and render pages, rejecting rendering over budget."""
    import fitz

    with fitz.open() as document:
        page = document.new_page(width=200, height=200)
        page.insert_text((20, 40), 'Supplementary table: 2.5')
        content = document.tobytes()
    part = readers.supplement_parts(asset(content, 'SI.pdf'))[0]
    assert 'Supplementary table: 2.5' in readers.supplement_text(part)
    images = readers.supplement_images(part, str(tmp_path / 'pdf'), dpi=72)
    assert len(images) == 1
    assert Image.open(images[0]).size == (200, 200)
    assert not (tmp_path / 'pdf' / 'source.pdf').exists()
    monkeypatch.setattr(readers, 'MAX_RENDERED_PIXELS', 10)
    with pytest.raises(ValueError, match='pixel limit'):
        readers.supplement_images(part, str(tmp_path / 'over-budget'), dpi=72)
    monkeypatch.setattr(readers, 'MAX_RENDERED_PIXELS', 100000)
    stream = io.BytesIO()
    Image.new('RGB', (3, 2), color='red').save(stream, format='TIFF')
    image_part = readers.supplement_parts(asset(stream.getvalue(), 'SI.tiff'))[0]
    assert image_part.supports_images and not image_part.supports_text
    rendered = readers.supplement_images(image_part, str(tmp_path / 'raster'))
    assert Image.open(rendered[0]).size == (3, 2)


def test_archive_skips_directories_and_refuses_encrypted_or_empty_archives() -> None:
    """Return only file members, and fail on archives that hold nothing readable."""
    payload = archive_bytes({'tables/': b'', 'tables/data.csv': b'Name\nLLZO'})
    parts = readers.supplement_parts(asset(payload, 'SI.zip'))
    assert [part.member for part in parts] == ['tables/data.csv']
    assert readers.supplement_text(parts[0]) == 'Name\nLLZO'

    with pytest.raises(ValueError, match='contains no files'):
        readers.supplement_parts(asset(archive_bytes({}), 'SI.zip'))

    encrypted = bytearray(archive_bytes({'secret.txt': b'LLZO, 1e-3 S/cm'}))
    # Mark the member encrypted in the central directory, which the reader trusts.
    encrypted[encrypted.rindex(b'PK\x01\x02') + 8] |= 1
    with pytest.raises(ValueError, match='Encrypted supplementary archives are unsupported'):
        readers.supplement_parts(asset(bytes(encrypted), 'SI.zip'))


def test_text_reader_decodes_utf16_and_windows_1252_exports() -> None:
    """Read UTF-16 files that start with a byte-order mark and legacy Windows-1252 text."""
    utf16 = readers.supplement_parts(asset('LLZO, 1.2 mS/cm'.encode('utf-16'), 'SI.csv'))[0]
    assert readers.supplement_text(utf16) == 'LLZO, 1.2 mS/cm'
    # 0xB0 is the degree sign in Windows-1252 and an invalid byte in UTF-8.
    legacy = readers.supplement_parts(asset(b'Sintered at 1100 \xb0C', 'SI.txt'))[0]
    assert readers.supplement_text(legacy) == 'Sintered at 1100 °C'


def test_xlsx_reader_names_sheets_from_workbook_relationships() -> None:
    """Label sheets by workbook name for relative and absolute targets, else by path."""
    sheet = (b'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
             b'<row r="1"><c r="A1"><v>%s</v></c></row></sheetData></worksheet>')
    xlsx = archive_bytes({
        'xl/workbook.xml': (
            b'<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            b'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'
            b'<sheet name="Conductivity" sheetId="1" r:id="rId1"/>'
            b'<sheet name="Synthesis" sheetId="2" r:id="rId2"/></sheets></workbook>'
        ),
        'xl/_rels/workbook.xml.rels': (
            b'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            b'<Relationship Id="rId1" Target="worksheets/sheet1.xml"/>'
            b'<Relationship Id="rId2" Target="/xl/worksheets/sheet2.xml"/></Relationships>'
        ),
        'xl/worksheets/sheet1.xml': sheet % b'1.2',
        'xl/worksheets/sheet2.xml': sheet % b'1100',
        'xl/worksheets/sheet3.xml': sheet % b'7',
    })

    text = readers.supplement_text(readers.supplement_parts(asset(xlsx, 'SI.xlsx'))[0])

    assert text.splitlines() == [
        'Sheet: Conductivity', 'A1: 1.2',
        'Sheet: Synthesis', 'A1: 1100',
        'Sheet: xl/worksheets/sheet3.xml', 'A1: 7',
    ]


def test_readers_refuse_parts_of_the_wrong_kind(tmp_path: Path) -> None:
    """Refuse text from an image and images from a text file instead of guessing."""
    stream = io.BytesIO()
    Image.new('RGB', (3, 2)).save(stream, format='PNG')
    image = readers.supplement_parts(asset(stream.getvalue(), 'figure.png'))[0]
    with pytest.raises(ValueError, match='Unsupported supplementary text format: figure.png'):
        readers.supplement_text(image)
    text = readers.supplement_parts(asset(b'LLZO, 1e-3 S/cm', 'notes.txt'))[0]
    with pytest.raises(ValueError, match='Unsupported supplementary image format: notes.txt'):
        readers.supplement_images(text, str(tmp_path / 'text'))
    assert [path for path in tmp_path.rglob('*') if path.is_file()] == []


def test_image_rendering_refuses_bad_dpi_and_inputs_over_page_frame_or_pixel_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail before rendering anything when a supplement exceeds a rendering limit."""
    import fitz

    with fitz.open() as document:
        for _ in range(2):
            document.new_page(width=100, height=100)
        pdf = readers.supplement_parts(asset(document.tobytes(), 'SI.pdf'))[0]
    with pytest.raises(ValueError, match='DPI must be positive'):
        readers.supplement_images(pdf, str(tmp_path / 'dpi'), dpi=0)
    monkeypatch.setattr(readers, 'MAX_PDF_PAGES', 1)
    with pytest.raises(ValueError, match='exceeds the 1 page rendering limit'):
        readers.supplement_images(pdf, str(tmp_path / 'pages'), dpi=72)

    stream = io.BytesIO()
    Image.new('RGB', (3, 2), color='red').save(
        stream, format='TIFF', save_all=True, append_images=[Image.new('RGB', (3, 2), color='blue')])
    frames = readers.supplement_parts(asset(stream.getvalue(), 'SI.tiff'))[0]
    monkeypatch.setattr(readers, 'MAX_ARCHIVE_ENTRIES', 1)
    with pytest.raises(ValueError, match='supported frame count'):
        readers.supplement_images(frames, str(tmp_path / 'frames'))
    monkeypatch.setattr(readers, 'MAX_ARCHIVE_ENTRIES', 2)
    monkeypatch.setattr(readers, 'MAX_RENDERED_PIXELS', 5)
    with pytest.raises(ValueError, match='total rendered pixel limit'):
        readers.supplement_images(frames, str(tmp_path / 'pixels'))
    assert [path for path in tmp_path.rglob('*') if path.is_file()] == []


@pytest.fixture
def fake_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use deterministic offline model and chunking substitutes."""
    monkeypatch.setattr(scrape, 'load_recipe', lambda recipe: sample_recipe())
    monkeypatch.setattr(scrape, 'ModelConfig', FakeModelConfig)
    monkeypatch.setattr(scrape, 'tqdm', FakeTqdm)
    monkeypatch.setattr(scrape, '_text_chunks', lambda text, model_config, prompt='': [text])
    monkeypatch.setattr(scrape, 'maybe_compress_text', lambda text, prompt, model_config, config: text)


def test_si_runs_after_article_success_with_file_provenance_and_independent_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_models: None,
) -> None:
    """Enable SI after a completed article, process every file, and checkpoint it."""
    db_path, output = tmp_path / 'papers.db', tmp_path / 'out.csv'
    paper = {'paper_id': 'paper-1', 'text_scrape_status': 'succeeded'}
    with corpus.connect(db_path) as conn:
        add_supplement(conn, paper, b'LLZO supporting table', original_filename='one.csv',
                       metadata={'label': 'Table S1', 'source_url': 'https://example.org/one.csv'})
        add_supplement(conn, paper, b'LATP method details', original_filename='two.txt')
    calls = []

    def extract(text: str, *_: object, **__: object) -> list[dict[str, str]]:
        """Record the model input and return one extracted record."""
        calls.append(text)
        return [{'Name': 'LLZO' if 'LLZO' in text else 'LATP'}]

    monkeypatch.setattr(scrape, 'scrape_text', extract)
    options = {'mode': 'text', 'output_path': str(output)}
    scrape.scrape_papers(str(db_path), include_supplements=False, **options)
    assert calls == []
    scrape.scrape_papers(str(db_path), **options)
    assert len(calls) == 2
    assert all('Supporting information:' in text and 'corpus:supplement:' in text for text in calls)
    rows = pd.read_csv(output)
    assert set(rows['Source']) == {'supplement-text'}
    assert set(rows['Supplement filename']) == {'one.csv', 'two.txt'}
    assert rows.loc[rows['Supplement filename'] == 'one.csv', 'Supplement source URL'].item() == 'https://example.org/one.csv'
    assert rows['Source path'].str.startswith('corpus:supplement:').all()
    with corpus.connect(db_path) as conn:
        persisted = corpus.paper_rows(conn)[0]
        assert persisted['text_scrape_status'] == 'succeeded'
        assert persisted['supplement_text_scrape_status'] == 'succeeded'
        assert all(item['metadata']['text_extraction_status'] == 'succeeded' for item in get_supplements(conn, paper['paper_id']))
    monkeypatch.setattr(scrape, 'supplement_parts', lambda _: pytest.fail('Completed SI must not be reparsed'))
    scrape.scrape_papers(str(db_path), **options)
    assert len(calls) == 2


def test_si_archive_partial_failure_retries_only_failed_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_models: None, capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep successful archive rows, retry a failed member, and report unsupported files."""
    db_path, output = tmp_path / 'papers.db', tmp_path / 'out.csv'
    paper = {'paper_id': 'paper-1', 'text_scrape_status': 'succeeded'}
    payload = archive_bytes({'good.txt': b'GOOD', 'retry.txt': b'RETRY', 'raw.xyz': b'raw source'})
    with corpus.connect(db_path) as conn:
        add_supplement(conn, paper, payload, original_filename='SI.zip')
    failures = [True]
    calls = []

    def extract(text: str, *_: object, **__: object) -> list[dict[str, str]]:
        """Simulate one transient extraction failure."""
        key = 'retry' if 'RETRY' in text else 'good'
        calls.append(key)
        if key == 'retry' and failures:
            failures.pop()
            raise RuntimeError('model unavailable')
        return [{'Name': key}]

    monkeypatch.setattr(scrape, 'scrape_text', extract)
    options = {'mode': 'text', 'output_path': str(output)}
    scrape.scrape_papers(str(db_path), **options)
    assert pd.read_csv(output)['Name'].tolist() == ['good']
    with corpus.connect(db_path) as conn:
        assert corpus.paper_rows(conn)[0]['supplement_text_scrape_status'] == 'partial'
        metadata = get_supplements(conn, paper['paper_id'])[0]['metadata']
        assert metadata['text_extraction_members'] == {'good.txt': 'succeeded', 'retry.txt': 'failed', 'raw.xyz': 'unsupported'}
    first_output = capsys.readouterr()
    assert 'failed file stages=1' in first_output.out
    assert 'unsupported file stages=1' in first_output.out
    assert 'model unavailable' in first_output.err
    scrape.scrape_papers(str(db_path), **options)
    assert calls == ['good', 'retry', 'retry']
    rows = pd.read_csv(output)
    assert rows['Name'].tolist() == ['good', 'retry']
    assert rows['Source path'].str.contains('!').all()
    with corpus.connect(db_path) as conn:
        assert get_supplements(conn, paper['paper_id'])[0]['content'] == payload


def test_si_needing_too_many_text_requests_is_unsupported_without_model_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_models: None,
) -> None:
    """Refuse a data-dump supplement before it costs one model request per chunk."""
    db_path, output = tmp_path / 'papers.db', tmp_path / 'out.csv'
    paper = {'paper_id': 'paper-1', 'text_scrape_status': 'succeeded'}
    with corpus.connect(db_path) as conn:
        add_supplement(conn, paper, b'0123\t4567\n' * 50, original_filename='genome.csv')
    monkeypatch.setattr(scrape, 'MAX_SUPPLEMENT_TEXT_REQUESTS', 3)
    monkeypatch.setattr(scrape, '_text_chunks', lambda text, model_config, prompt='': [text] * 4)
    monkeypatch.setattr(scrape, 'scrape_text',
                        lambda *_args, **_kwargs: pytest.fail('an oversized supplement must not reach the model'))

    scrape.scrape_papers(str(db_path), mode='text', output_path=str(output))

    with corpus.connect(db_path) as conn:
        metadata = get_supplements(conn, paper['paper_id'])[0]['metadata']
    assert metadata['text_extraction_status'] == 'unsupported'
    assert 'genome.csv: too large for text extraction (4 requests, limit 3)' in metadata['text_extraction_error']
    assert not output.exists()


def test_si_split_into_overlapping_chunks_merges_repeated_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_models: None,
) -> None:
    """Report a record once when the chunk overlap shows it to the model twice."""
    db_path, output = tmp_path / 'papers.db', tmp_path / 'out.csv'
    paper = {'paper_id': 'paper-1', 'text_scrape_status': 'succeeded'}
    with corpus.connect(db_path) as conn:
        add_supplement(conn, paper, b'LLZO conducts at 1 mS/cm.\n', original_filename='table.txt')
    monkeypatch.setattr(scrape, '_text_chunks', lambda text, model_config, prompt='': ['part one', 'part two'])
    monkeypatch.setattr(scrape, 'scrape_text', lambda *_args, **_kwargs: [{'Name': 'LLZO'}])

    scrape.scrape_papers(str(db_path), mode='text', output_path=str(output))

    assert pd.read_csv(output)['Name'].tolist() == ['LLZO']


def test_si_image_mode_supplies_labels_and_preserves_stable_asset_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_models: None,
) -> None:
    """Send labelled SI images and keep provenance after image cleanup."""
    db_path, output = tmp_path / 'papers.db', tmp_path / 'out.csv'
    paper = {'paper_id': 'paper-1', 'image_scrape_status': 'succeeded'}
    stream = io.BytesIO()
    Image.new('RGB', (3, 2)).save(stream, format='PNG')
    with corpus.connect(db_path) as conn:
        blob_id = add_supplement(conn, paper, stream.getvalue(), original_filename='figure.png',
                                 metadata={'label': 'Figure S4'})
    captured = []

    def extract(paths: list[str], *_: object, **kwargs: Any) -> list[dict[str, str]]:
        """Inspect the vision input and return an extracted record."""
        assert all(Path(path).is_file() for path in paths)
        captured.append((paths, kwargs))
        return [{'Name': 'LLZO'}]

    monkeypatch.setattr(scrape, 'scrape_images', extract)
    scrape.scrape_papers(str(db_path), mode='images', output_path=str(output),
                        image_dir=str(tmp_path / 'images'), delete_images_after=True)
    assert len(captured) == 1
    assert 'Figure S4' in captured[0][1]['image_labels'][0]
    assert f'corpus:supplement:{blob_id}' in captured[0][1]['image_labels'][0]
    assert all(not Path(path).exists() for path in captured[0][0])
    rows = pd.read_csv(output)
    assert rows['Source'].tolist() == ['supplement-image']
    assert rows['Source path'].tolist() == [f'corpus:supplement:{blob_id}']


def test_abstract_mode_never_loads_supplements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_models: None,
) -> None:
    """Leave supporting information out of abstract-only extraction."""
    db_path = tmp_path / 'papers.db'
    with corpus.connect(db_path) as conn:
        corpus.upsert_paper(conn, {'paper_id': 'one', 'abstract_scrape_status': 'succeeded'})
    monkeypatch.setattr(scrape, 'get_supplements', lambda *args, **kwargs: pytest.fail('Abstract scrape loaded SI'))
    scrape.scrape_papers(str(db_path), mode='abstract', output_path=str(tmp_path / 'out.csv'))


def test_si_duplicate_blobs_are_processed_once_and_force_reprocesses_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_models: None,
) -> None:
    """Avoid duplicate output from identical blobs linked with different sources."""
    db_path, output = tmp_path / 'papers.db', tmp_path / 'out.csv'
    paper = {'paper_id': 'one', 'text_scrape_status': 'succeeded'}
    with corpus.connect(db_path) as conn:
        blob_id = add_supplement(conn, paper, b'same supplementary content', original_filename='SI.txt')
        corpus.link_asset(conn, paper['paper_id'], blob_id, role='supplement', source='mirror',
                          original_filename='mirror.txt')
    calls = []

    def extract(text: str, *_: object, **__: object) -> list[dict[str, str]]:
        """Return one record per requested SI input."""
        calls.append(text)
        return [{'Name': 'one'}]

    monkeypatch.setattr(scrape, 'scrape_text', extract)
    options = {'mode': 'text', 'output_path': str(output)}
    scrape.scrape_papers(str(db_path), **options)
    assert len(calls) == 1
    scrape.scrape_papers(str(db_path), **options)
    assert len(calls) == 1
    # --force also retries the absent article, but independently reruns SI only once.
    scrape.scrape_papers(str(db_path), force=True, **options)
    assert len(calls) == 2
    assert len(pd.read_csv(output)) == 2


def test_si_text_images_has_independent_stages_and_retains_corrupt_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_models: None,
) -> None:
    """Run both SI stages independently and record corrupt archive errors."""
    import fitz

    db_path, output = tmp_path / 'papers.db', tmp_path / 'out.csv'
    paper = {'paper_id': 'one', 'text_scrape_status': 'succeeded', 'image_scrape_status': 'succeeded'}
    with fitz.open() as document:
        document.new_page(width=100, height=100).insert_text((5, 20), 'SI data')
        content = document.tobytes()
    with corpus.connect(db_path) as conn:
        add_supplement(conn, paper, content, original_filename='SI.pdf')
        add_supplement(conn, paper, b'corrupt zip', original_filename='bad.zip')
    monkeypatch.setattr(scrape, 'scrape_text', lambda *args, **kwargs: [{'Name': 'text result'}])
    monkeypatch.setattr(scrape, 'scrape_images', lambda *args, **kwargs: [{'Name': 'image result'}])
    monkeypatch.setattr(scrape, 'combine_material_records', lambda *args, **kwargs: pytest.fail('SI provenance was combined'))
    scrape.scrape_papers(str(db_path), mode='text-images', output_path=str(output),
                        image_dir=str(tmp_path / 'images'), image_context='paper-text', image_dpi=72)
    rows = pd.read_csv(output)
    assert rows['Source'].tolist() == ['supplement-text', 'supplement-image']
    with corpus.connect(db_path) as conn:
        stored = {entry['original_filename']: entry for entry in get_supplements(conn, paper['paper_id'])}
        assert stored['bad.zip']['content'] == b'corrupt zip'
        for stage in ('text', 'image'):
            assert stored['bad.zip']['metadata'][f'{stage}_extraction_status'] == 'failed'
            assert stored['SI.pdf']['metadata'][f'{stage}_extraction_status'] == 'succeeded'
            assert corpus.paper_rows(conn)[0][f'supplement_{stage}_scrape_status'] == 'partial'


def test_si_without_content_for_a_stage_is_not_applicable_and_never_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_models: None,
) -> None:
    """Record a figure-only supplement as not applicable to text, without a model call."""
    db_path, output = tmp_path / 'papers.db', tmp_path / 'out.csv'
    paper = {'paper_id': 'paper-1', 'text_scrape_status': 'succeeded'}
    stream = io.BytesIO()
    Image.new('RGB', (3, 2)).save(stream, format='PNG')
    with closing(corpus.connect(db_path)) as conn, conn:
        add_supplement(conn, paper, stream.getvalue(), original_filename='figure.png')
    monkeypatch.setattr(scrape, 'scrape_text',
                        lambda *_args, **_kwargs: pytest.fail('an image supplement has no text to send'))
    options = {'mode': 'text', 'output_path': str(output)}

    scrape.scrape_papers(str(db_path), **options)

    with closing(corpus.connect(db_path)) as conn, conn:
        metadata = get_supplements(conn, paper['paper_id'])[0]['metadata']
        assert corpus.paper_rows(conn)[0]['supplement_text_scrape_status'] == 'not_applicable'
    assert metadata['text_extraction_status'] == 'not_applicable'
    assert metadata['text_extraction_members'] == {'figure.png': 'not_applicable'}
    assert metadata['text_extraction_error'] == ''
    assert not output.exists()
    monkeypatch.setattr(scrape, 'supplement_parts',
                        lambda _: pytest.fail('a not-applicable SI stage must not be reparsed'))
    scrape.scrape_papers(str(db_path), **options)
    with closing(corpus.connect(db_path)) as conn, conn:
        assert corpus.paper_rows(conn)[0]['supplement_text_scrape_status'] == 'not_applicable'


def test_si_pdf_without_a_text_layer_fails_and_points_to_image_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_models: None,
) -> None:
    """Fail a scanned SI PDF's text stage with a hint instead of sending an empty request."""
    import fitz

    db_path, output = tmp_path / 'papers.db', tmp_path / 'out.csv'
    paper = {'paper_id': 'paper-1', 'text_scrape_status': 'succeeded'}
    with fitz.open() as document:
        document.new_page(width=100, height=100)
        content = document.tobytes()
    with closing(corpus.connect(db_path)) as conn, conn:
        add_supplement(conn, paper, content, original_filename='scan.pdf')
    monkeypatch.setattr(scrape, 'scrape_text',
                        lambda *_args, **_kwargs: pytest.fail('an empty supplement must not reach the model'))

    scrape.scrape_papers(str(db_path), mode='text', output_path=str(output))

    with closing(corpus.connect(db_path)) as conn, conn:
        metadata = get_supplements(conn, paper['paper_id'])[0]['metadata']
        assert corpus.paper_rows(conn)[0]['supplement_text_scrape_status'] == 'failed'
    assert metadata['text_extraction_status'] == 'failed'
    assert metadata['text_extraction_members'] == {'scan.pdf': 'failed'}
    assert metadata['text_extraction_error'] == (
        'scan.pdf: No extractable text; use image mode for scanned PDF pages')
    assert not output.exists()


def test_si_image_stage_fails_when_a_supplement_yields_no_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_models: None,
) -> None:
    """Fail the image stage of a PDF with no pages rather than calling the vision model."""
    from pypdf import PdfWriter

    db_path, output = tmp_path / 'papers.db', tmp_path / 'out.csv'
    paper = {'paper_id': 'paper-1', 'image_scrape_status': 'succeeded'}
    stream = io.BytesIO()
    PdfWriter().write(stream)
    with closing(corpus.connect(db_path)) as conn, conn:
        add_supplement(conn, paper, stream.getvalue(), original_filename='empty.pdf')
    monkeypatch.setattr(scrape, 'scrape_images',
                        lambda *_args, **_kwargs: pytest.fail('no images means no vision request'))

    scrape.scrape_papers(str(db_path), mode='images', output_path=str(output),
                         image_dir=str(tmp_path / 'images'))

    with closing(corpus.connect(db_path)) as conn, conn:
        metadata = get_supplements(conn, paper['paper_id'])[0]['metadata']
        assert corpus.paper_rows(conn)[0]['supplement_image_scrape_status'] == 'failed'
    assert metadata['image_extraction_status'] == 'failed'
    assert metadata['image_extraction_members'] == {'empty.pdf': 'failed'}
    assert metadata['image_extraction_error'] == (
        'empty.pdf: No supplementary images or PDF pages could be read')
    assert not output.exists()

def test_archive_member_that_reads_past_its_declared_size_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse a member whose data outgrows the size limit even when its header declared less."""
    payload = archive_bytes({'data.txt': b'1234'})
    monkeypatch.setattr(readers, 'MAX_MEMBER_BYTES', 4)

    def oversized(self: zipfile.ZipFile, entry: zipfile.ZipInfo) -> io.BytesIO:
        """Yield more bytes than the member header declares, as a lying archive reader could."""
        return io.BytesIO(b'12345')

    monkeypatch.setattr(zipfile.ZipFile, 'open', oversized)
    with pytest.raises(ValueError, match='member exceeds the size limit'):
        readers.supplement_parts(asset(payload, 'SI.zip'))
