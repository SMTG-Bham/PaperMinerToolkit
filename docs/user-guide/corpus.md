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
ps_search "Lithium solid electrolyte" papers.db --source pubmed --count 100
```

PubMed exposes only the first 10000 matches for any query, whatever `--count` asks for. When a
search matches more than that, PaperScraper prints the shortfall; split the query by date range to
reach the rest.

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

## Supplement metadata

Search and import records carry only a handful of bibliographic fields. `ps_enrich` fills in the rest
from Crossref, OpenAlex, and PubMed:

```bash
ps_enrich papers.db
```

Crossref supplies the metadata the publisher deposited against the DOI — publisher, work type,
volume, issue, pages, ISSNs and language. OpenAlex supplies what it derives — citation counts,
open-access status and licence, retraction flags, and subject classification. Structured authors
(with ORCID and institution ROR), subjects, and reference lists are written to the `paper_authors`,
`paper_subjects`, and `paper_references` tables.

PubMed supplies the National Library of Medicine's controlled vocabulary for papers that carry a
PMID: MeSH descriptors and qualifiers, publication types, and author keywords, written to
`paper_subjects` under the `mesh`, `mesh_qualifier`, `publication_type`, and `mesh_keyword` schemes.
Each provider's child rows are replaced independently, so enriching from one source never discards
another's.

Enrichment is re-runnable and resumable. A second run costs nothing because it only selects papers
that are still pending, and an interrupted run keeps everything it already committed:

```bash
ps_enrich papers.db --limit 500        # stop after 500 papers, resume later
ps_enrich papers.db --retry-failed     # retry only papers that previously failed
ps_enrich papers.db --refresh-after 90 # refresh citation counts older than 90 days
ps_enrich papers.db --force            # re-fetch everything
```

Restrict the providers, or skip reference lists when you only want the bibliographic fields:

```bash
ps_enrich papers.db --source crossref
ps_enrich papers.db --source openalex --no-references
ps_enrich papers.db --source pubmed
```

Papers with no DOI, OpenAlex identifier, or PMID are reported as skipped and cost no requests. A
row that carries only a PMID — common for records found through PubMed itself — is resolvable
whenever PubMed is among the selected sources.
To supplement rows as they arrive instead of in a separate pass, add `--enrich` to discovery:

```bash
ps_search "Lithium solid electrolyte" papers.db --enrich
ps_import_author supervisor.db --orcid 0000-0000-0000-0000 --enrich
```

`ps_reset` re-arms the enrichment stage without discarding enrichment data, so a reset corpus can be
re-enriched without refetching everything.

## Download abstracts, text, and PDFs

```bash
ps_download papers.db --format abstract
ps_download papers.db --format text
ps_download papers.db --format pdf
ps_download papers.db --format both
```

Abstract retrieval tries OpenAlex, PubMed, CORE, and Elsevier in that order; PubMed is attempted for
any row carrying a PMID or a DOI. PDF retrieval can use Unpaywall, OpenAlex, CORE, Elsevier, and
PubMed Central, which is tried last. Select PDF sources by repeating `--source`:

```bash
ps_download papers.db --format pdf \
  --source unpaywall \
  --source openalex
```

PaperScraper tracks abstracts, full text, and PDFs independently. It skips a requested type when that type is already present while continuing to obtain missing types. Override this protection only when deliberately refreshing content:

```bash
ps_download papers.db --format both --force
```

Full text comes from Elsevier when a key is configured, and otherwise from the PubMed Central
open-access subset, which needs no credentials:

```bash
ps_download papers.db --format text --source pubmed
```

Only the PMC open-access subset is redistributable, so a paper with a PMC identifier outside that
subset reports that no full text is offered rather than failing. An NCBI API key raises the PubMed
request rate from three to ten per second, which matters most on large download runs; see
[Credentials and model configuration](../getting-started/configuration.md).

By default, abstracts are also attempted while downloading text or PDFs. Disable that with `--no-abstract`.

## Inspect the corpus

```bash
ps_corpus_stats papers.db
ps_status papers.db
```

Corpus statistics include paper and asset counts, original and compressed storage sizes, and counts of text or abstract inputs that required chunking. Pipeline status summarizes search, download, scrape, and storage progress.

The database is the durable source of truth. CSV files produced by scraping, prediction, trends, and storage are exports or intermediate artifacts and can be recreated from the corpus and saved models.
