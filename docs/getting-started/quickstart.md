# Quickstart

This example searches for a small corpus, downloads abstracts, scrapes one recipe, and stores the records. Configure at least one search source and a text model first.

## 1. Search

```bash
pm_search "lithium solid electrolyte" papers.db --source openalex --count 25
pm_corpus_stats papers.db
```

## 2. Download content

Start with abstracts for a cheap smoke test:

```bash
pm_download papers.db --format abstract
```

For full extraction, request text and PDFs. Already stored content is skipped:

```bash
pm_download papers.db --format both
```

## 3. Scrape a recipe

```bash
pm_scrape papers.db sse --mode text --output temp_scraped_materials.csv
```

The recipe may be a bundled name such as `sse`, `polymer`, `polymer_db`, or `band_gap_validation`, or a path to your own JSON recipe.

## 4. Store the results

```bash
pm_store \
  papers.db \
  temp_scraped_materials.csv \
  materials.csv \
  sse \
  --assume-yes

pm_status papers.db
```

Use the same recipe for scraping and storage so fields and unit conversions agree. Continue with {doc}`../user-guide/corpus`, {doc}`../user-guide/filtering`, and {doc}`../user-guide/scraping` for production workflows.
