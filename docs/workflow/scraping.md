# Scraping and storing records

A recipe defines what one output record represents, the extraction instructions, output fields, examples, aliases, and unit conversions. See {doc}`recipes` for the complete recipe format and an explanation of how PaperMinerToolkit constructs prompts. Use exactly the same recipe for scraping and storage.

## Bundled recipes

`sse`
: Lithium-conducting solid electrolytes, including composition, structure, conductivity, and electrochemical properties.

`polymer`
: Polymer identity and architecture, printed line notations, thermal and mechanical properties, molecular weight, and biodegradation results.

`polymer_db`
: A wider polymer-database schema with identifiers, composition, microstructure, solution properties, degradation curves, and OECD test metadata. Its large output schema is best suited to papers with relatively few distinct records.

`band_gap_validation`
: One record per distinct material, sample, composition, phase, or structure. Principal results, every gap reported by the study, and cited literature gaps are kept in separate JSON lists.

Pass a bundled name or an external JSON path:

```bash
pmt scrape papers.db sse --mode text
pmt scrape papers.db ./my_recipe.json --mode text
```

Store each recipe in its own final CSV. Existing output columns participate in alias matching, so unrelated recipe schemas should not share a file.

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

Successful stages are skipped by default. Force a deliberate rescrape:

```bash
pmt scrape papers.db sse --mode text --force
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

The general missing value is the string `None`. Recipes that define list-valued output may use an empty list where the whole list is supported but contains no items; follow each recipe's field-level prompt.

## Review against a validation set

Run `pmt validate` to open a local browser review. Choose a validation CSV, a
scraped CSV, and the recipe used for extraction. The recipe selector lists
bundled recipes and **Custom recipe JSON…**, which reveals a file picker for a
standalone recipe JSON file. For repeat runs, provide all three paths on the
command line:

```bash
pmt validate \
  --validation tests/data/verification_data/band_gaps.csv \
  --scraped temp_scraped_materials.csv \
  --recipe band_gap_validation
```

The reviewer groups rows by DOI, then paper ID or title when needed. It suggests
material pairs within each paper; confirm or change pairs, mark missing and extra
entries, and record a correct or incorrect decision for each recipe field.
Per-entry and per-paper progress is shown in the browser. Decisions autosave to
`~/.config/paperminertoolkit/validation-reviews/` and resume when the same two
CSVs and recipe are selected again. The setup screen also lists saved reviews;
it can resume them from local CSV snapshots without asking you to find the files
again. These snapshots stay in the same local review folder. The input CSVs are
never changed. Use `--output review.json` to choose the decision file.

For list-valued fields, pair items and judge each scalar value separately.
The reviewer splits structured CSV cells written as JSON or Python-style
list/dictionary literals, while the **Raw cell** view retains the original text.
Dictionary-valued answers expose each key for its own decision, including keys
inside list items such as `value`, `method_or_source`, `gap_type`, and
`conditions`. Nested lists and dictionaries use the same controls; mark missing
reference parts and unpaired extracted parts extra. The scoring summary shows
both field totals and expandable per-part totals.

An extra scraped entry, list item, or dictionary part can be marked correct or
incorrect. Correct extras count as true positives when the reference omitted
valid information; incorrect extras count as false positives. Unreviewed extras
keep the score pending.

Record the extraction mode, model identifier, provider, context length,
software commit, run date, and the numeric matching rule in **Scoring summary**.
That view gives per-field TP, FP, FN, precision, recall, and F1. While a review
is in progress, metrics use the decisions completed so far and remain
provisional; an undefined denominator is reported as pending until the review
is complete, then as null. A wrong matched value counts one FP and one FN, a missing
value counts one FN. Use **Download scoring
run** to archive the decisions, metrics, protocol, paper count, and SHA-256
fingerprints of both CSVs and the recipe. A download made before review is
complete is labeled a draft.
