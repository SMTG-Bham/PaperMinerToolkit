"""Validate PDF figure detection against the licensed publisher test fixture."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from PIL import Image

from paperminertoolkit.corpus.pdf_layout import detect_pdf_layout, render_pdf_figures

PDF = Path(__file__).parent / 'data' / 'disorder-driven_fast_na_transport_oxychlorides.pdf'
PDF_SHA256 = 'ff8bd821fb5f8762f0295eef8b20ca8e376a943b259948ee2efb467482d55ccb'
# Page, display label, and caption anchor checked against the publisher PDF.
GOLD_FIGURES = (
    (3, 'Figure 1', 'Weighted experimental pair distribution function'),
    (4, 'Figure 2', 'Normalized XANES spectra'),
    (5, 'Figure 3', 'Two-dimensional representation of the SOAP vectors'),
    (6, 'Figure 4', 'Room-temperature ionic conductivities'),
    (7, 'Figure 5', 'Representative charge'),
    (8, 'Figure 6', 'retained discharge capacity'),
    (9, 'Figure 7', 'Representative motifs'),
    (9, 'Figure 8', 'Representative motifs'),
    (10, 'Figure 9', 'diffusion coefficient'),
)


def _normalise(text: str) -> str:
    """Ignore case, punctuation, and PDF line wrapping in caption anchors."""
    return ' '.join(re.findall(r'[a-z0-9]+', text.casefold()))


def test_pdf_figure_extraction_matches_gold_labels(tmp_path: Path) -> None:
    """Find and render every publisher figure, with no extra figures."""
    assert hashlib.sha256(PDF.read_bytes()).hexdigest() == PDF_SHA256

    layout = detect_pdf_layout(PDF)
    detected = [(figure.page_numbers[0], figure.label.casefold()) for figure in layout.figures]
    expected = [(page, label.casefold()) for page, label, _ in GOLD_FIGURES]
    assert sorted(detected) == sorted(expected)

    by_key = {
        (figure.page_numbers[0], figure.label.casefold()): figure
        for figure in layout.figures
    }
    for page, label, caption_anchor in GOLD_FIGURES:
        figure = by_key[(page, label.casefold())]
        assert _normalise(caption_anchor) in _normalise(figure.caption)
        assert figure.boxes, f'{label} on page {page} has no visual region'

    rendered = render_pdf_figures(PDF, layout, tmp_path, dpi=72)
    assert len(rendered) == len(GOLD_FIGURES)
    for path in rendered:
        with Image.open(path) as image:
            assert image.width >= 20 and image.height >= 20
            assert image.convert('L').getextrema()[0] <= 245