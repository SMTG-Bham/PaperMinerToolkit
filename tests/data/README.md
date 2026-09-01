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
