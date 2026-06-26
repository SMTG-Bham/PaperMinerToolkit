import pandas as pd

from paperscraper.pipeline import PAPER_COLUMNS, PIPELINE_COLUMNS, merge_paper_rows, normalize_paper_columns


def test_merge_paper_rows_deduplicates_by_doi_and_keeps_schema():
    existing = pd.DataFrame([
        {
            'paper_id': 'elsevier:1',
            'doi': '10.1234/ABC',
            'title': 'A Solid Electrolyte Paper',
            'publication_date': '2024-01-01',
            'sources': 'elsevier',
            'elsevier_link': 'full-text-link',
        }
    ])
    incoming = pd.DataFrame([
        {
            'paper_id': 'core:99',
            'doi': 'https://doi.org/10.1234/abc.',
            'title': 'A Solid Electrolyte Paper',
            'publication_date': '2024',
            'sources': 'core',
            'core_id': '99',
            'pdf_url': 'https://example.test/paper.pdf',
        }
    ])

    merged, added, updated = merge_paper_rows(existing, incoming)

    assert added == 0
    assert updated == 1
    assert len(merged) == 1
    assert merged.loc[0, 'sources'] == 'elsevier;core'
    assert merged.loc[0, 'core_id'] == '99'
    assert merged.loc[0, 'pdf_url'] == 'https://example.test/paper.pdf'
    assert list(merged.columns) == PAPER_COLUMNS + list(PIPELINE_COLUMNS)


def test_normalize_paper_columns_drops_unknown_columns_and_adds_defaults():
    raw = pd.DataFrame([
        {
            'paper_id': 'paper:1',
            'doi': '10.1234/test',
            'extra_provider_field': 'not kept',
        }
    ])

    normalized = normalize_paper_columns(raw)

    assert 'extra_provider_field' not in normalized.columns
    assert normalized.loc[0, 'metadata_status'] == 'pending'
    assert normalized.loc[0, 'num_images'] == 0
    assert list(normalized.columns) == PAPER_COLUMNS + list(PIPELINE_COLUMNS)
