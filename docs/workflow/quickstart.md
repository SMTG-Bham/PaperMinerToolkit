# Quickstart

This example gathers a small corpus of abstracts, scrapes one recipe, and stores
the records. Probing and gathering from arXiv require no API key or language model;
configure a text model only when you reach extraction.

## 1. Probe and gather

```bash
pmt probe --source arxiv
pmt gather "lithium solid electrolyte" papers.db \
  --source arxiv --count 5 --format abstract --download-source arxiv
pmt corpus stats papers.db
```

`gather` searches and merges paper records, stores abstracts returned by the
search, and fetches missing content for just the papers matched in this run.
Repeat `--source` to search several providers, and `--download-source` to select
content providers separately. Configure only the {doc}`credentials <configuration>`
those providers need. Add `--json` to `probe` or `gather` for machine-readable
standard output; progress is sent to standard error.

## 2. Add content when needed

The abstracts are ready for extraction. To collect PDFs as well, run:

```bash
pmt download papers.db --format pdf --source arxiv
```

Already stored content is skipped. For future queries, gather abstracts and PDFs
together with `--format pdf`. The default `--format both` requests text and PDFs
from the selected content providers; arXiv itself serves abstracts and PDFs only.
You can also run discovery and downloading separately:

```bash
pmt search "lithium solid electrolyte" papers.db --source arxiv --count 5
pmt download papers.db --format abstract --source arxiv
```

## 3. Scrape a recipe

```bash
pmt config model text --provider openai --model YOUR_TEXT_MODEL
pmt scrape papers.db sse --mode abstract --output temp_scraped_materials.csv
```

The recipe may be a bundled name such as `sse`, `polymer`, `polymer_db`, or `band_gap_validation`, or a path to your own JSON recipe.

## 4. Store the results

```bash
pmt store \
  papers.db \
  temp_scraped_materials.csv \
  materials.csv \
  sse \
  --assume-yes

pmt status papers.db
```

Use the same recipe for scraping and storage so fields and unit conversions agree. Continue with {doc}`corpus`, {doc}`filtering`, and {doc}`scraping` for production workflows.
