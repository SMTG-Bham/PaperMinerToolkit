# Scraping and storing records

A recipe defines what one output record represents, the extraction instructions, output fields, examples, aliases, and unit conversions. See {doc}`recipes` for the complete recipe format and an explanation of how PaperMinerToolkit constructs prompts. Use exactly the same recipe for scraping and storage.

## Bundled recipes

`sse`
: Lithium-conducting solid electrolytes, including composition, structure, conductivity, and electrochemical properties.

`polymer`
: Polymer identity and architecture, printed line notations, thermal and mechanical properties, molecular weight, and biodegradation results. Distinct samples and separate degradation tests produce separate records.

`polymer_db`
: A wider polymer-database schema with identifiers, composition, microstructure, solution properties, degradation time series, and OECD test metadata, using the same separation of samples and tests. Its larger schema leaves less completion space for additional records.

`band_gap_validation`
: One record per distinct material, sample, composition, phase, or structure. Principal results, every gap reported by the study, and cited literature gaps are kept in separate JSON lists.

Pass a bundled name or an external JSON path:

```bash
pmt scrape papers.db sse --mode text
pmt scrape papers.db ./my_recipe.json --mode text
```

Store each recipe in its own final CSV. Existing output columns participate in alias matching, so unrelated recipe schemas should not share a file.

Both polymer recipes copy structure identifiers only when the source prints them; they do not infer SMILES or BigSMILES from polymer names or drawings.

Scalar biodegradation and mass-loss percentages describe the end of the test, paired with `Biodegradation test duration`. An unreported endpoint is `None`, even if an earlier maximum is available. `polymer_db` also preserves reported time series in its kinetics and degradation-curve fields. The example records describe one hypothetical homopolymer sample in one compost test; their values are not evidence to copy into an extraction.

Biodegradability claims must be stated for the sample and conditions. Bio-based origin alone does not establish biodegradability, as explained by the [European Environment Agency](https://www.eea.europa.eu/en/analysis/publications/biodegradable-and-compostable-plastics-challenges-and-opportunities). Report the paper's measured biodegradation endpoint: [OECD 301](https://www.oecd.org/en/publications/test-no-301-ready-biodegradability_9789264070349-en.html), for example, includes methods based on dissolved organic carbon, CO2 production, and oxygen uptake. Keep these results separate from gravimetric mass loss and molecular-weight loss.

## Text and image modes

Text only:

```bash
pmt scrape papers.db sse --mode text
```

Images only:

```bash
pmt scrape papers.db sse --mode images
```

Combined text and images:

```bash
pmt scrape papers.db sse \
  --mode text-images \
  --image-context paper-text
```

When both modes produce records, the text model reconciles matching records into `text+image` rows. Use `--image-batch-size N` or `--image-batch-size all` only when the vision model has enough context capacity.

### Choosing which images the model sees

`--image-extraction` selects where images come from:

- `layout` sends real figures with their captions. Structured figures already downloaded into the corpus are used first; a paper with none has its figures detected from PDF geometry and stored in the corpus, so a later run reuses them instead of detecting again. A paper with no figures from either route fails rather than falling back.
- `auto` (the default) prefers those layout-aware figures, and uses embedded PDF images or rendered pages when a paper has none and none can be detected.
- `embedded` extracts every raster image the PDF contains, including logos and decorative art.
- `pages` renders one image per page.

```bash
pmt scrape papers.db sse --mode images --image-extraction layout
```

Layout mode changes what reaches the model and what comes back. Each image is preceded by its figure label and caption, so the model can attribute a value to a specific figure, and every extracted row records `Figure id`, `Figure label`, and `Figure source` alongside the existing `Source path`. Because a structured figure and its PDF-rendered equivalent would otherwise be analysed twice, detection runs only for papers whose corpus holds no figures yet.

Progress is checkpointed per figure rather than per paper: a figure analysed successfully is skipped on the next run, a figure whose request failed is retried, and `--force` reanalyses every figure. An interrupted run therefore resumes without paying for the figures it already processed.

### Detecting figures directly

The Python API exposes the same detection used by layout mode. `detect_pdf_layout` and
`render_pdf_figures` in `paperminertoolkit.corpus.pdf_layout` detect `Figure`, `Fig.`, and `Table`
captions, join wrapped captions within a column, and associate them with nearby raster or vector
geometry. Confident figure regions are rendered with configurable padding and resolution, clamped
so a crop never includes neighbouring caption text; uncertain associations render the complete
source page. Panel detection is not performed.

`paperminertoolkit.workflows.figures.store_pdf_layout_figures` wraps that detection and writes the
results into the corpus as figure assets, which is what layout mode calls. Use
`render_pdf_figures` directly when image files on disk are wanted instead.

## Context limits and compression

PaperMinerToolkit reserves space for prompts and output before sending source content. Optional compression can reduce oversized text or image inputs. If text still exceeds the usable model context, it is split into independent requests.

:::{warning}
Records extracted from separate chunks are not reconciled automatically. A material spanning chunk boundaries may be duplicated or incomplete. Increase the configured input limit only when the serving model genuinely supports it.
:::

The corpus records `num_text_chunks` and `num_abstract_chunks`. A value of `1` means the input fit one request; larger values indicate splitting. Inspect aggregate counts with `pmt corpus stats`.

## Reruns and temporary files

Successful stages are skipped by default. Statuses belong to the corpus and stage, not to a recipe, so selecting another recipe does not rerun successful stages. Use `--force` when changing recipes or deliberately rescraping, and choose a fresh intermediate output path because scrape output appends to existing files:

```bash
pmt scrape papers.db polymer_db --mode text --force --output scraped_polymer_db_rerun.csv
```

Choose the intermediate output and remove extracted images after successful analysis when scratch space matters:

```bash
pmt scrape papers.db sse \
  --mode images \
  --output scraped_materials.csv \
  --delete-images-after
```

## Aggregate and store results

```bash
pmt store \
  papers.db \
  temp_scraped_materials.csv \
  materials.csv \
  sse \
  --assume-yes
```

Storage matches aliases, performs recipe-defined unit conversions, appends provenance metadata, merges the new rows with the final CSV, and records stored papers in the corpus. Review the intermediate file before omitting confirmation in unattended workflows.

Unit conversion batches values by unit-bearing column. All-missing columns need no model request; large populated columns may require several. A recipe's count of unit-bearing fields is not a fixed request count.

Conversion can remove conditions embedded in unit-bearing values. Keep a copy of the scraped CSV before storage if you need its original wording; successful storage consumes the input file. To store without conversion, use the Python API:

```python
from paperminertoolkit.extraction.store import store_results

store_results(
    db_path="papers.db",
    in_filepath="scraped_polymer_db_rerun.csv",
    out_filepath="materials_polymer_db_rerun.csv",
    recipe="polymer_db",
    unit_conversion=False,
    assume_yes=True,
)
```

With conversion disabled, column headings still carry the recipe's target-unit labels, but cell values retain their original units and wording.

The general missing value is the string `None`. Recipes that define list-valued output may use an empty list where the whole list is supported but contains no items; follow each recipe's field-level prompt.

## Review against a validation set

Create a blank validation CSV from the recipe, then open the local review GUI:

```bash
pmt validate template band_gap_validation band_gap_validation.csv
pmt validate gui \
  --validation band_gap_validation.csv \
  --scraped temp_scraped_materials.csv \
  --recipe band_gap_validation
```

The {doc}`validation` guide explains the CSV format, paper and record pairing,
structured-field review, autosave and resume behavior, scoring rules, required
run metadata, and exported scoring archive.
