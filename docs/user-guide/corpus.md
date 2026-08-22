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
ps_search "Lithium solid electrolyte" papers.db --source arxiv --count 100
ps_search "Lithium solid electrolyte" papers.db --source medrxiv --count 100
```

PubMed exposes only the first 10000 matches for any query, whatever `--count` asks for. When a
search matches more than that, PaperScraper prints the shortfall; split the query by date range to
reach the rest. arXiv has the same limit at 30000 matches, and the same shortfall message.

arXiv accepts a fielded query language rather than a plain phrase. A plain phrase is translated
for you — its words are combined with `AND` across all fields — but a query that already uses a
field prefix or a boolean operator is sent as written, so you can be precise:

```bash
ps_search 'cat:cond-mat.mtrl-sci AND abs:"solid electrolyte"' papers.db --source arxiv
ps_search 'ti:garnet ANDNOT au:Smith' papers.db --source arxiv
```

The available prefixes are `ti:`, `au:`, `abs:`, `co:`, `jr:`, `cat:`, `rn:`, `id:`, and `all:`.

arXiv is a preprint server, so its records often describe a paper the corpus already holds.
When the submitting author deposited a DOI, the arXiv row merges into the published row, which
keeps its own journal and date and simply gains the arXiv identifier and PDF location. Without a
deposited DOI the only remaining rule is a matching title and year, so a preprint posted in one
calendar year and published in the next is stored as its own row. arXiv is searched last for
exactly this reason: the published record wins whenever the two do match.

medRxiv publishes no search endpoint. Its API can return one preprint by DOI, or every posting in
a date range, and nothing else. PaperScraper answers a medRxiv query by reading that archive
newest first and matching each posting itself, over the title, abstract, authors, and category. A
term matches at the start of a word, so `vaccine` finds `vaccines` and `covid` finds `covid-19`,
and several terms are combined with `AND`.

That makes the query the only place to say how much should be read, so it carries its own scope.
`category:`, `from:`, and `to:` narrow the archive; everything else is a match term:

```bash
ps_search 'vaccine hesitancy category:"public and global health"' papers.db --source medrxiv
ps_search '"long covid" from:2024-01-01 to:2024-06-30' papers.db --source medrxiv
```

Without them the walk starts at today and runs back to the first medRxiv posting in June 2019.
Before it starts, PaperScraper prints how many postings that is. The walk ends as soon as
`--count` papers match, so a query about medical research is usually answered in a few requests;
a query that matches nothing recent is the expensive case, and it stops after 20000 postings and
says how many it left unread. Narrowing with `category:`, `from:`, or `to:` is what reaches the
rest.

Categories are medRxiv's own list, written as they appear on the site: `infectious diseases`,
`epidemiology`, `public and global health`, `psychiatry and clinical psychology`, and so on.
Quote any that contain a space.

Like arXiv, a medRxiv preprint that has since been published is stored under the published DOI
so it merges with the published row, and keeps its preprint DOI in `medrxiv_doi`. A preprint with
no published version is stored under its own DOI with `medRxiv` as the journal.

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
from Crossref, OpenAlex, PubMed, arXiv, and medRxiv:

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
arXiv supplies its own subject taxonomy for papers that carry an arXiv identifier, written to
`paper_subjects` under the `arxiv_category` scheme with the submission's primary category
flagged. It also backfills an author-deposited DOI, which is what lets a preprint-only row reach
Crossref and OpenAlex on a later pass, and records that the paper is freely readable. arXiv ranks
last, because its record describes the preprint rather than the version of record, so it only
fills fields no other provider supplied.
medRxiv supplies its subject category under the `medrxiv_category` scheme, the licence the
authors posted the preprint under, and the link between a preprint DOI and the published DOI. It
ranks below arXiv for the same reason and one more: a medRxiv row that names a published version
already carries that version's DOI, so Crossref and OpenAlex describe the paper better than the
preprint record can.

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
ps_enrich papers.db --source arxiv
ps_enrich papers.db --source medrxiv
```

Papers with no DOI, OpenAlex identifier, PMID, arXiv identifier, or medRxiv DOI are reported as
skipped and cost no requests. A row that carries only one of those identifiers — common for
records found through PubMed, arXiv, or medRxiv themselves — is resolvable whenever the provider
that knows it is among the selected sources. Note that neither arXiv nor medRxiv can be reached
from an ordinary DOI: arXiv publishes no DOI search field at all, and a published medRxiv row
carries the journal's DOI rather than the preprint's. Both enrich only rows that already carry
the identifier they issued.
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

Abstract retrieval tries OpenAlex, PubMed, medRxiv, arXiv, CORE, and Elsevier in that order;
PubMed is attempted for any row carrying a PMID or a DOI, and medRxiv and arXiv for any row
carrying the identifier each issued. PDF retrieval can use Unpaywall, OpenAlex, CORE, Elsevier,
PubMed Central, medRxiv, and arXiv, in that order. The two preprint servers are tried last
because the other sources may hold the publisher's version of record while a preprint server
holds the preprint, which is a different document. Select PDF sources by repeating `--source`:

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

arXiv serves PDFs and abstracts but no full text, because it publishes no machine-readable
full-text format. Text for an arXiv paper comes from scraping its downloaded PDF.

medRxiv is the other way round. Every posting names a JATS document, the same format PubMed
Central serves, so `--format text` takes medRxiv full text directly rather than scraping a PDF.
Its PDFs sit behind a bot challenge that occasionally refuses a client outright; when that
happens the run reports the refusal for that paper and carries on, and the text source is
unaffected.

By default, abstracts are also attempted while downloading text or PDFs. Disable that with `--no-abstract`.

## Inspect the corpus

```bash
ps_corpus_stats papers.db
ps_status papers.db
```

Corpus statistics include paper and asset counts, original and compressed storage sizes, and counts of text or abstract inputs that required chunking. Pipeline status summarizes search, download, scrape, and storage progress.

The database is the durable source of truth. CSV files produced by scraping, prediction, trends, and storage are exports or intermediate artifacts and can be recreated from the corpus and saved models.
