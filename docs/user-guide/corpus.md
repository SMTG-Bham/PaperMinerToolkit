# Building a corpus

A PaperScraper corpus is a SQLite database containing normalized paper metadata, compressed content-addressed assets, pipeline state, filter decisions, and optional topic-model predictions. Matching DOI, source identifier, or title/year records are merged so a paper is not processed twice.

## Search literature services

Search all configured providers:

```bash
ps_search "Lithium solid electrolyte" papers.db --count 200
```

Select a single provider when you need reproducible source coverage:

```bash
ps_search "Lithium solid electrolyte" papers.db --source core --count 100
ps_search "Lithium solid electrolyte" papers.db --source openalex --count 100
ps_search "Lithium solid electrolyte" papers.db --source elsevier --count 100
```

`--count` is applied to each selected provider. Add `--store-abstract` to retain abstracts returned in search records immediately; otherwise abstracts can be fetched during downloading.

## Import local PDFs

Import every PDF in a directory:

```bash
ps_import_pdfs papers/ papers.db
```

The importer scans PDF metadata and text for DOI candidates, looks up Crossref metadata, and stores the original PDF in the corpus. For offline imports, retain DOI extraction but skip Crossref:

```bash
ps_import_pdfs papers/ papers.db --no-crossref
```

## Import an author's works

Crossref can seed a corpus before any content is downloaded. Prefer an ORCID:

```bash
ps_import_author supervisor.db \
  --orcid 0000-0000-0000-0000 \
  --email you@example.ac.uk \
  --review-csv supervisor_works.csv
```

If no ORCID is available, use a full name and optionally an affiliation:

```bash
ps_import_author supervisor.db \
  --author "First Family" \
  --affiliation "University of Example" \
  --email you@example.ac.uk
```

Inspect the review CSV before downloading. Crossref provides metadata and DOIs, not guaranteed access to paper content.

## Download abstracts, text, and PDFs

```bash
ps_download papers.db --format abstract
ps_download papers.db --format text
ps_download papers.db --format pdf
ps_download papers.db --format both
```

Abstract retrieval tries OpenAlex, CORE, and Elsevier. PDF retrieval can use Unpaywall, OpenAlex, CORE, and Elsevier. Select PDF sources by repeating `--source`:

```bash
ps_download papers.db --format pdf \
  --source unpaywall \
  --source openalex
```

PaperScraper tracks abstracts, full text, and PDFs independently. It skips a requested type when that type is already present while continuing to obtain missing types. Override this protection only when deliberately refreshing content:

```bash
ps_download papers.db --format both --force
```

By default, abstracts are also attempted while downloading text or PDFs. Disable that with `--no-abstract`.

## Inspect the corpus

```bash
ps_corpus_stats papers.db
ps_status papers.db
```

Corpus statistics include paper and asset counts, original and compressed storage sizes, and counts of text or abstract inputs that required chunking. Pipeline status summarizes search, download, scrape, and storage progress.

The database is the durable source of truth. CSV files produced by scraping, prediction, trends, and storage are exports or intermediate artifacts and can be recreated from the corpus and saved models.
