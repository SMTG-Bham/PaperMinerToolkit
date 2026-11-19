# Scraping and storing records

A recipe defines what one output record represents, the extraction instructions, output fields, examples, aliases, and unit conversions. See {doc}`recipes` for the complete recipe format and an explanation of how PaperMinerToolkit constructs prompts. Use exactly the same recipe for scraping and storage.

## Bundled recipes

`sse`
: Lithium-conducting solid electrolytes, including composition, structure, conductivity, and electrochemical properties.

`polymer`
: Polymer identity and architecture, printed line notations, thermal and mechanical properties, molecular weight, and biodegradation results. Distinct samples and separate degradation tests produce separate records.

`polymer_db`
: A wider polymer-database schema with identifiers, composition, microstructure, solution properties, degradation time series, and OECD test metadata, using the same separation of samples and tests. Its larger schema leaves less completion space for additional records.

`polymer_degradation`
: Biotic and abiotic degradation experiments: composting, soil, marine, anaerobic, enzymatic and microbial tests alongside hydrolysis, photodegradation, thermo-oxidative ageing, and weathering. Each record is one sample in one test, with its conditions, controls, endpoint values, and complete mass loss and mineralisation curves. Sterile controls and each stage of a sequential test are separate records, and a plotted mass loss curve is identified by its figure so it can be digitised later.

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

### Tables

Full-text scrapes append the paper's tables to the text, under a `=== TABLES ===` heading, as Markdown grids with each table's label, caption and footnotes. Plain-text assets are built without tables, so they are taken from the paper's stored structured document (JATS, Elsevier XML or TEI) when it has one; otherwise ruled tables are detected in the PDF with PyMuPDF. Row and column spans are expanded, so each value sits in its own row and column. Use `--no-tables` to send the text alone.

When both modes produce records, the text model reconciles matching records into `text+image` rows. Use `--image-batch-size N` or `--image-batch-size all` only when the vision model has enough context capacity.

### Supporting information

Stored supporting information is processed by default in `text`, `images`, and
`text-images` modes. Abstract-only runs do not process SI. Acquisition and
extraction are separate: `scrape` reads stored assets, while `download` or
`gather` discovers and retrieves them.

```bash
pmt download papers.db --format supplement
pmt scrape papers.db sse --mode text-images --output extracted_with_si.csv

# Exclude stored SI from a particular scrape run
pmt scrape papers.db sse --mode text --no-supplements
```

Text extraction supports PDF, plain text, CSV, TSV, JSON, XML, HTML, DOCX text
and tables, and XLSX cached cell values. PDF pages and supported raster images
can be sent to the vision model. ZIP members are processed in memory within
entry-count, expanded-size, and compression-ratio limits; nested ZIP archives
are not expanded. Unsupported formats and processing errors are reported, and
the original stored files are retained.

Each SI file or archive member is processed separately. Extracted rows carry
`supplement-text` or `supplement-image` provenance, the asset identity, and
available source URL and label. SI records are not automatically reconciled
with article records or other SI files, so a repeated material can produce
several records with different evidence sources.

A supporting-information file whose text would need more than 20 requests, such
as a genome annotation or instrument export, is marked unsupported with the
request count instead of being sent.

Successful text and image stages are checkpointed per SI asset. A later run
processes new or failed SI even when the main article was already scraped.
`--force` repeats successful SI stages; `--no-supplements` leaves SI checkpoints
unchanged. As with article scraping, use `--force` and a fresh output path when
changing recipes. Paper-level SI text and image status fields summarize these
separate stages.

(choosing-which-images-the-model-sees)=
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

The same steps are available as commands, so figures can be prepared and checked
before a paid image scrape:

| Command | What it does |
| --- | --- |
| `pmt figures download papers.db` | Downloads the figure images referenced by stored JATS, Elsevier XML, or GROBID TEI documents. They replace any PDF-detected figures of the same paper. |
| `pmt figures detect papers.db` | Detects figures in stored PDFs and stores the crops, for papers without structured figures. `--force` replaces earlier detections; `--dpi`, `--padding`, and `--min-confidence` tune them. |
| `pmt figures list papers.db` | Lists stored figures with source, caption, whether a region was found, and scrape and digitization status. |
| `pmt figures export papers.db outdir` | Writes stored figures to `outdir/<paper>/<figure>.<ext>`. |
| `pmt figures render paper.pdf outdir --strategy layout` | Runs `layout`, `embedded`, or `pages` extraction on any PDF, without a corpus. |

Each paper keeps figures from one origin, so `layout` mode and `pmt digitize` never see a
figure twice. All five commands accept `--json`; `download` and `detect` also take
`--paper-id` and follow active filters.

## Context limits and compression

PaperMinerToolkit reserves space for prompts and output before sending source content. Optional compression can reduce oversized text or image inputs. If text still exceeds the usable model context, it is split into independent requests. Chunks are cut between paragraphs and never inside a table (an oversized table is split between rows, repeating its header), and each chunk repeats the end of the previous paragraph. A chunk that still measures over the limit, such as one dense with numbers, is split again.

The records from all chunks of a paper (or of one supporting-information file split into several requests) are then merged without another model call: records whose identity fields match (ignoring case, punctuation and word order) become one record, list items that state the same value are kept once, and other fields keep their first reported value. A record that only repeats an earlier record's values under a qualified version of its name, such as `Cs3Bi2I9 (thin film)` after `Cs3Bi2I9 (powder)`, is dropped.

Tokens are counted with the configured model's own tokenizer: tiktoken for OpenAI models, the token-counting endpoint for Anthropic, and the Hugging Face tokenizer named by the model setting for local servers. Every chunk is measured with that tokenizer, and one that still overflows is split again using its own token density, because density is not uniform: a table of numbers can need three times the tokens of the same length of prose. When a tokenizer cannot be loaded, counts fall back to an estimate of one token per three characters and a warning names the tokenizer that failed. Treat that warning as a fault to fix (for a local server, the model name must be its exact Hugging Face repository ID): the estimate can undercount numeric text badly enough to overflow the context.

:::{warning}
Merging is deterministic. A record whose identity is worded differently in two chunks, or whose details are split across chunks, can still be duplicated or incomplete. Increase the configured input limit when the serving model genuinely supports it.
:::

The corpus records `num_text_chunks` and `num_abstract_chunks`. A value of `1` means the input fit one request; larger values indicate splitting. Inspect aggregate counts with `pmt corpus stats`.

## Preview a run with --dry-run

`pmt scrape ... --dry-run` reports what the same command would do without
calling a model or changing the corpus. It selects papers exactly as the real
run would (filters, `--paper-id`, `--order`, `--count`, `--force`), reads their
stored text as the run would (with its `--tables` and `--figure-text` settings),
and splits it with the same chunker and token budget. It reports,
for each paper and each supporting-information file:

- whether the stage would run, be skipped as already done, or fail for lack of
  a downloaded source;
- the number of model requests and the estimated input tokens;
- inputs that need more than one request, and SI files over the
  20-request limit, which a real run skips as unsupported;
- whether compression would apply. Token counts are taken before compression.

```bash
pmt scrape papers.db sse --mode text --dry-run
pmt scrape papers.db sse --mode text --dry-run --json > plan.json
```

Token counting is offline. OpenAI models use `tiktoken`'s cached encoding, and
local models use their Hugging Face tokenizer from local files only. Anthropic
models fall back to the character estimate, because their exact count needs an
API request. Any fallback is listed in `tokenizer_fallbacks`. Image inputs are
counted but not extracted or measured. The corpus is opened read-only, so an
older corpus must be migrated first by any corpus-writing command, such as
`pmt status`.

## Checking records against the paper

A recipe can declare `checks` (see [Recipes](recipes.md), under "Checks"). They run only when asked for:

```bash
pmt scrape papers.db band_gap_validation --mode text --verify --output scraped.csv
```

With `--verify`, each paper's records are checked against its full text, including tables. Items whose numbers are not printed in the paper, or that break the recipe's other checks, are removed. Every removed or edited item is listed, with the check and the reason, in `<output stem>.dropped.csv` (here `scraped.dropped.csv`).

To measure the effect on a scrape that has already run, apply the same merging and checks to its CSV:

```bash
pmt postprocess papers.db band_gap_validation scraped.csv
```

This writes `scraped.postprocessed.csv` and `scraped.postprocessed.dropped.csv`. The source text is read from the corpus as it would be for a scrape. Image records are kept unchecked. Use `--no-merge`, `--no-verify` or `--no-tables` to apply only part of it.

## Concurrent requests

By default one paper is scraped at a time. A model server that batches requests, such as vLLM, sits mostly idle that way. With `--workers N`, abstract and text scrapes keep up to N papers' requests in flight:

```bash
pmt scrape papers.db band_gap_validation --mode text --verify --workers 8
```

Papers are still read from the corpus, and their rows, statuses and audit entries are still written, one at a time in scrape order, so the output is the same as a sequential run's. On one A100-80GB serving gpt-oss-120b, 8 workers cut a 113-paper scrape from 29 to 13 minutes. Choose N so the server's KV cache holds that many requests (vLLM logs its usage). Hosted APIs may rate-limit concurrent requests, and image modes always run one request at a time.

## Text printed inside figures

A PDF's text layer includes the labels and numbers printed inside its figures: axis ticks, legends, and values in heat maps or annotated plots. In text mode these reach the model as loose numbers that it can mistake for reported values. With `--no-figure-text`, figures are detected as in layout mode (art matched to a caption), and short text blocks inside them are redacted before the PDF is read:

```bash
pmt scrape papers.db band_gap_validation --mode text --verify --no-figure-text
```

Captions, tables and body prose are kept. A block with a line of 40 or more characters is treated as prose even inside a figure region, and figure regions that overlap a captioned table are left alone. Papers read from a publisher text asset are unaffected. The option also applies to `pmt postprocess`, so values shown only in a figure fail the values-in-source check. To extract values from figures, use an image mode.

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

Supporting-information asset IDs, filenames, labels, source URLs, archive
member references, and the `Value warnings` that record values replaced during
extraction remain in the final CSV, including when appending to an older
output file. Separate SI files remain separate evidence rows. Papers with only
successful SI results can also be marked as stored.

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

The review GUI labels supporting-information entries in the pairing selector
and displays their source details beside the selected record and in unpaired
entry cards. Provenance is retained separately from the recipe fields being
scored.
