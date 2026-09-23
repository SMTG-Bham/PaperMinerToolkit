"""PDF-only figure validation against two CC BY 4.0 Scanlon papers.

Full-caption fingerprints are SHA-256 of lowercase ASCII alphanumeric tokens joined
by single spaces. Gold captions were reviewed on the fixed PDF pages; fingerprints
avoid duplicating article prose in the test. A mismatch fails the validation test.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import fitz
import pytest
from PIL import Image

from paperminertoolkit.corpus.pdf_layout import detect_pdf_layout, render_pdf_figures

DATA = Path(__file__).parent / 'data'
# (page, label, caption anchor, full-caption fingerprint)
WILEY_GOLD = (
    (3, 'Figure 1', 'Weighted experimental pair distribution function', '055a95daed4f37e3bcca567fd86aad70aef61fb69a0535c913c91e8ec31bb22d'),
    (4, 'Figure 2', 'Normalized XANES spectra', '4792657caf2d54380d06652895a1196f6dc05a13d9a6deec28d890f30df6191e'),
    (5, 'Figure 3', 'Two-dimensional representation of the SOAP vectors', 'c7bfeb1cb393b59fc53d87b318d1e14d4f3a598fe76d060f92176af54d349efa'),
    (6, 'Figure 4', 'Room-temperature ionic conductivities', '4fbbd18fe366bc962ed92da9dcb76dfa773aefdfdcabe08951b64cd34524cb31'),
    (7, 'Figure 5', 'Representative charge', '28202e0f0bff7b09507f9fa5b038a63840ef67319b887f3ced4d33ede940e9d1'),
    (8, 'Figure 6', 'retained discharge capacity', '78dad71d15ca49259032091cd57a8d2bf2850f7fb28e5f801177128fdebbd2df'),
    (9, 'Figure 7', 'Representative motifs', '5f1b36f0adad49db46ab738d5b211ac23deaf93f8651fd22cb024eeba23a6f89'),
    (9, 'Figure 8', 'Representative motifs', '96605c99e5f2ba2022d11254f2ece2cf025e3bae4b75cc32dab84fb592d1ff46'),
    (10, 'Figure 9', 'diffusion coefficient', 'b65102100f8d5ed13ee89788502de169235e868ce8af1cf99da5c7dfe6bb3f12'),
)
SCANLON_GOLD = (
    # The source caption ends at "initial configuration (Dq)".
    (2, 'Fig. 1', 'Graph network illustration', '52b6aadf364d86762ceff5760b24c04be1628079d75108a2f77d88fed8c959c9'),
    (6, 'Fig. S2', 'Minimum energy paths', 'f8451fb0b91d10b0ff3c7305a80cdd481fe4a477671383a5304dc0cd02a652c8'),
)
FIXTURES = (
    ('disorder-driven_fast_na_transport_oxychlorides.pdf', 'ff8bd821fb5f8762f0295eef8b20ca8e376a943b259948ee2efb467482d55ccb', WILEY_GOLD),
    ('scanlon_metastable_excerpt.pdf', 'aedd2a6ede84e0ec12e8bd69be65bcedf97574309252db8b1ab65433283b6c04', SCANLON_GOLD),
)


def _normalise(text: str) -> str:
    return ' '.join(re.findall(r'[a-z0-9]+', text.casefold()))


def _caption_fingerprint(text: str) -> str:
    return hashlib.sha256(_normalise(text).encode()).hexdigest()


@pytest.mark.parametrize('filename,pdf_sha256,gold', FIXTURES)
def test_pdf_figure_validation(
    filename: str,
    pdf_sha256: str,
    gold: tuple[tuple[int, str, str, str], ...],
    tmp_path: Path,
) -> None:
    """Measure recall, full-caption association, false positives, and crops."""
    pdf = DATA / filename
    assert hashlib.sha256(pdf.read_bytes()).hexdigest() == pdf_sha256
    layout = detect_pdf_layout(pdf)
    by_key = {(f.page_numbers[0], f.label.casefold()): f for f in layout.figures}
    gold_keys = {(page, label.casefold()) for page, label, _, _ in gold}
    matched = gold_keys & by_key.keys()
    recall = len(matched) / len(gold)
    false_positives = len(layout.figures) - len(matched)
    caption_matches = sum(
        _caption_fingerprint(by_key[(page, label.casefold())].caption) == fingerprint
        for page, label, _, fingerprint in gold
        if (page, label.casefold()) in matched
    )
    caption_association_accuracy = caption_matches / len(gold)
    metrics = (f'{filename}: recall={recall:.1%}, '
               f'caption association={caption_association_accuracy:.1%}, '
               f'false positives={false_positives}')

    assert recall == 1, metrics
    assert false_positives == 0, metrics
    assert caption_matches == len(gold), metrics
    for page, label, anchor, _ in gold:
        figure = by_key[(page, label.casefold())]
        assert _normalise(anchor) in _normalise(figure.caption), metrics
        assert figure.boxes, f'{metrics}; {label} has no visual region'

    with fitz.open(pdf) as document:
        if filename == 'scanlon_metastable_excerpt.pdf':
            # Raster, table, and vector examples are on distinct excerpt pages.
            assert document[1].get_images(full=True)
            assert not document[5].get_images(full=True)
            assert len(document[5].get_drawings()) >= 20
            assert any(table.page_numbers == (4,) for table in layout.tables)
            assert all(f.page_numbers != (4,) for f in layout.figures)
        else:
            # The publisher logo is decorative art, not a reported figure.
            assert document[0].get_images(full=True)
            assert all(f.page_numbers != (1,) for f in layout.figures)

    rendered = render_pdf_figures(pdf, layout, tmp_path, dpi=72)
    assert len(rendered) == len(gold), metrics
    for path in rendered:
        with Image.open(path) as image:
            assert image.width >= 20 and image.height >= 20
            assert image.convert('L').getextrema()[0] <= 245
