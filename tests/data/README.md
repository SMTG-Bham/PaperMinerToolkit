This folder contains the data used in the tests.

## Third-party test data

### `disorder-driven_fast_na_transport_oxychlorides.pdf`

Justin Leifeld et al., “Disorder-Driven Fast Na+ Transport: From
Crystalline to Amorphous Networks in the Mixed-Anion NaTaOxCl6−2x
Oxychlorides,” *Advanced Energy Materials* **16** (2026), e70977.

https://doi.org/10.1002/aenm.70977

© 2026 The Authors. Published by Wiley-VCH GmbH. Licensed under the
[Creative Commons Attribution 4.0 International licence](https://creativecommons.org/licenses/by/4.0/).

The unmodified publisher PDF is included as a PaperMinerToolkit test fixture. It is
not covered by the repository's MIT licence.

### `biorxiv_2023.03.30.534894v4.jats.xml`

Jiaxin Li et al., "Vangl2 suppresses NF-κB signaling and ameliorates sepsis by
targeting p65 for NDP52-mediated autophagic degradation," bioRxiv (2024).

https://doi.org/10.1101/2023.03.30.534894

Licensed under the
[Creative Commons Attribution 4.0 International licence](https://creativecommons.org/licenses/by/4.0/).

### `medrxiv_2024.05.31.24307874v1.jats.xml`

Smart Wristband Monitoring: A Caregiver-Oriented Approach, medRxiv (2024).

https://doi.org/10.1101/2024.05.31.24307874

Licensed under the
[Creative Commons Attribution 4.0 International licence](https://creativecommons.org/licenses/by/4.0/).

Both files are the archives' own JATS, trimmed to the article title and the
first few figure elements; every retained element is byte-for-byte as
published. They are included so the XML layout parsers are tested against
markup the archives really serve rather than a plausible imitation of it, and
the medRxiv one is kept because it contains a multi-panel figure. The licence
permits the modification that trimming represents, which is why CC BY articles
were chosen over the CC BY-NC-ND that most preprints carry. Neither file is
covered by the repository's MIT licence.

### `openalex_grobid_wrapped.tei.xml`

Not a third-party document. It is a hand-written skeleton that reproduces the
exact shape OpenAlex serves GROBID TEI in: the TEI wrapped in `<html><body>`
and every element name lower-cased, because the document has been through an
HTML serialiser. Both properties were confirmed against a real
`content.openalex.org` response, and both had to be handled before that
endpoint produced a layout at all. The structure is real; the prose is
invented, so no article text is redistributed here.

### `scanlon_metastable_excerpt.pdf`

Seán R. Kavanagh, David O. Scanlon, Aron Walsh, and Christoph Freysoldt,
“Impact of metastable defect structures on carrier recombination in solar
cells,” arXiv:2202.12212v2 (2022).

Source: https://arxiv.org/abs/2202.12212 and
https://arxiv.org/pdf/2202.12212v2. The arXiv record links the
[Creative Commons Attribution 4.0 International licence](https://creativecommons.org/licenses/by/4.0/).
© the authors. This PDF is not covered by the repository's MIT licence.

This is an adapted, six-page excerpt of the 8,039,091-byte source PDF, retaining
original PDF pages 1, 2, 3, 8, 16, and 17 in that order. The source PDF has
SHA-256 `895736135eee99ead1a19dde852305fe7f12cf928d824f28d10853eaac375793`;
the excerpt has SHA-256
`aedd2a6ede84e0ec12e8bd69be65bcedf97574309252db8b1ab65433283b6c04`.
CC BY 4.0 permits sharing an adapted excerpt with attribution and notice of the
change. The full source PDF is kept outside Git; only the smaller excerpt is a
test fixture.

The excerpt provides a single-column raster Figure 1 (excerpt page 2), a table
(page 4), and a vector supplemental Figure S2 (page 6). Pages 3 and 5 have
in-text figure references but no figure art, which tests false positives.
The Wiley PDF above supplies double-column raster figures and a decorative
publisher logo on page 1. The page numbers and full-caption fingerprints used
as gold labels are in `tests/corpus/test_pdf_figure_validation.py`.

Current measured baseline for the two fixed PDFs: 11/11 figure recall,
11/11 exact full-caption associations, and zero false positives. The Scanlon
excerpt specifically guards against in-text figure references being counted as
figures and against body text being appended to its Figure 1 caption.
