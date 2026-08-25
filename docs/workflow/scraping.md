# Scraping and storing records

A recipe defines what one output record represents, the extraction instructions, output fields, examples, aliases, and unit conversions. See {doc}`recipes` for the complete recipe format and an explanation of how PaperMiner constructs prompts. Use exactly the same recipe for scraping and storage.

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
pm scrape papers.db sse --mode text
pm scrape papers.db ./my_recipe.json --mode text
```

Store each recipe in its own final CSV. Existing output columns participate in alias matching, so unrelated recipe schemas should not share a file.

## Text and image modes

Text only:

```bash
pm scrape papers.db sse --mode text
```

Images only:

```bash
pm scrape papers.db sse --mode images
```

Combined text and images:

```bash
pm scrape papers.db sse \
  --mode text-images \
  --image-context paper-text
```

When both modes produce records, the text model reconciles matching records into `text+image` rows. Image extraction uses embedded PDF images when possible and otherwise renders pages. Override this with `--image-extraction embedded` or `--image-extraction pages`. Use `--image-batch-size N` or `--image-batch-size all` only when the vision model has enough context capacity.

## Context limits and compression

PaperMiner reserves space for prompts and output before sending source content. Optional compression can reduce oversized text or image inputs. If text still exceeds the usable model context, it is split into independent requests.

:::{warning}
Records extracted from separate chunks are not reconciled automatically. A material spanning chunk boundaries may be duplicated or incomplete. Increase the configured input limit only when the serving model genuinely supports it.
:::

The corpus records `num_text_chunks` and `num_abstract_chunks`. A value of `1` means the input fit one request; larger values indicate splitting. Inspect aggregate counts with `pm corpus stats`.

## Reruns and temporary files

Successful stages are skipped by default. Force a deliberate rescrape:

```bash
pm scrape papers.db sse --mode text --force
```

Choose the intermediate output and remove extracted images after successful analysis when scratch space matters:

```bash
pm scrape papers.db sse \
  --mode images \
  --output scraped_materials.csv \
  --delete-images-after
```

## Aggregate and store results

```bash
pm store \
  papers.db \
  temp_scraped_materials.csv \
  materials.csv \
  sse \
  --assume-yes
```

Storage matches aliases, performs recipe-defined unit conversions, appends provenance metadata, merges the new rows with the final CSV, and records stored papers in the corpus. Review the intermediate file before omitting confirmation in unattended workflows.

The general missing value is the string `None`. Recipes that define list-valued output may use an empty list where the whole list is supported but contains no items; follow each recipe's field-level prompt.
