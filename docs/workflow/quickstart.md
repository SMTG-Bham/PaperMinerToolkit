# Quickstart

This example searches for a small corpus, downloads abstracts, scrapes one recipe, and stores the records. Configure at least one search source and a text model first.

## 1. Search

```bash
pmt search "lithium solid electrolyte" papers.db --source openalex --count 25
pmt corpus stats papers.db
```

## 2. Download content

Start with abstracts for a cheap smoke test:

```bash
pmt download papers.db --format abstract
```

For full extraction, request text and PDFs. Already stored content is skipped:

```bash
pmt download papers.db --format both
```

## 3. Scrape a recipe

```bash
pmt scrape papers.db sse --mode text --output temp_scraped_materials.csv
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
