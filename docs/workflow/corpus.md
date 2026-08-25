# Building a corpus

A PaperMinerToolkit corpus is a SQLite database containing normalized paper metadata, compressed content-addressed assets, pipeline state, filter decisions, and optional topic-model predictions. Matching DOI, source identifier, or title/year records are merged so a paper is not processed twice.

## Search literature services

Search all configured providers:

```bash
pmt search "Lithium solid electrolyte" papers.db --count 200
```

Select a single provider when you need reproducible source coverage:

```bash
pmt search "Lithium solid electrolyte" papers.db --source core --count 100
pmt search "Lithium solid electrolyte" papers.db --source openalex --count 100
pmt search "Lithium solid electrolyte" papers.db --source elsevier --count 100
pmt search "Lithium solid electrolyte" papers.db --source pubmed --count 100
pmt search "Lithium solid electrolyte" papers.db --source arxiv --count 100
pmt search "Lithium solid electrolyte" papers.db --source medrxiv --count 100
pmt search "Lithium solid electrolyte" papers.db --source biorxiv --count 100
pmt search "Lithium solid electrolyte" papers.db --source chemrxiv --count 100
```

PubMed exposes only the first 10000 matches for any query, whatever `--count` asks for. When a
search matches more than that, PaperMinerToolkit prints the shortfall; split the query by date range to
reach the rest. arXiv has the same limit at 30000 matches, and the same shortfall message.

arXiv accepts a fielded query language rather than a plain phrase. A plain phrase is translated
for you — its words are combined with `AND` across all fields — but a query that already uses a
field prefix or a boolean operator is sent as written, so you can be precise:

```bash
pmt search 'cat:cond-mat.mtrl-sci AND abs:"solid electrolyte"' papers.db --source arxiv
pmt search 'ti:garnet ANDNOT au:Smith' papers.db --source arxiv
```

The available prefixes are `ti:`, `au:`, `abs:`, `co:`, `jr:`, `cat:`, `rn:`, `id:`, and `all:`.

arXiv is a preprint server, so its records often describe a paper the corpus already holds.
When the submitting author deposited a DOI, the arXiv row merges into the published row, which
keeps its own journal and date and simply gains the arXiv identifier and PDF location. Without a
deposited DOI the only remaining rule is a matching title and year, so a preprint posted in one
calendar year and published in the next is stored as its own row. arXiv is searched last for
exactly this reason: the published record wins whenever the two do match.

medRxiv and bioRxiv publish no search endpoint. Their API can return one preprint by DOI, or
every posting in a date range, and nothing else. PaperMinerToolkit answers such a query by reading that
archive newest first and matching each posting itself, over the title, abstract, authors, and
category. A term matches at the start of a word, so `vaccine` finds `vaccines` and `genome` finds
`genomes`, and several terms are combined with `AND`.

That makes the query the only place to say how much should be read, so it carries its own scope.
`category:`, `from:`, and `to:` narrow the archive; everything else is a match term:

```bash
pmt search 'vaccine hesitancy category:"public and global health"' papers.db --source medrxiv
pmt search '"long covid" from:2024-01-01 to:2024-06-30' papers.db --source medrxiv
pmt search 'chromatin category:"developmental biology"' papers.db --source biorxiv
pmt search '"single cell" from:2024-01-01 to:2024-06-30' papers.db --source biorxiv
```

Without them the walk starts at today and runs back to the first posting in the archive: June 2019
for medRxiv, November 2013 for bioRxiv. Before it starts, PaperMinerToolkit prints how many postings
that is. The walk ends as soon as `--count` papers match, so a query about the archive's own
subject matter is usually answered in a few requests; a query that matches nothing recent is the
expensive case, and it stops after 20000 postings and says how many it left unread. Narrowing with
`category:`, `from:`, or `to:` is what reaches the rest.

Scoping matters more for bioRxiv, which is both older and larger, so an unscoped walk over it is
the slower of the two by a wide margin.

Categories are each server's own list, written as they appear on the site. medRxiv files under
`infectious diseases`, `epidemiology`, `public and global health`, `psychiatry and clinical
psychology`, and so on; bioRxiv under `neuroscience`, `bioinformatics`, `microbiology`, `cell
biology`, `developmental biology`, and the rest of its life-science list. Quote any that contain a
space.

Like arXiv, a preprint that has since been published is stored under the published DOI so it
merges with the published row, and keeps its preprint DOI in `medrxiv_doi` or `biorxiv_doi`. A
preprint with no published version is stored under its own DOI with `medRxiv` or `bioRxiv` as the
journal.

The two servers share a DOI prefix — `10.1101` for older postings and `10.64898` for newer ones —
so the prefix does not say which archive a preprint belongs to. PaperMinerToolkit tells them apart by
the accession number, which is six digits on bioRxiv and eight on medRxiv, and routes each row to
the one server that can answer for it. bioRxiv postings from before 2018 carry a bare accession
such as `10.1101/060400` instead, which is recognized too.

chemRxiv is the chemistry preprint archive, and it works differently from the other two. It does
publish a search endpoint, so a query is answered by the server rather than by reading the archive
and matching locally. The same `category:`, `from:`, and `to:` terms are accepted, but here they
are passed to the service as filters:

```bash
pmt search 'photocatalysis category:Catalysis' papers.db --source chemrxiv
pmt search '"metal organic framework" from:2024-01-01 to:2024-12-31' papers.db --source chemrxiv
```

Narrowing a chemRxiv query therefore makes the service do less work rather than saving you a long
read, and an unscoped query is not the expensive case it is for medRxiv and bioRxiv. Only the first
10000 matches are reachable for any one query; when a query matches more, PaperMinerToolkit prints the
shortfall, and a `category:` or date scope reaches the rest. Categories are chemRxiv's own list —
`Catalysis`, `Organic Chemistry`, `Analytical Chemistry`, `Theoretical and Computational
Chemistry`, and the rest — and a name that matches none of them is reported rather than ignored.

A chemRxiv DOI keeps the version it was issued with, and that suffix is part of the DOI rather than
decoration on the end of it. `10.26434/chemrxiv.15007737/v1` is a registered DOI while
`10.26434/chemrxiv.15007737` is not, and for the older dated accessions the reverse holds:
`10.26434/chemrxiv-2022-w08rh` is registered and `10.26434/chemrxiv-2022-w08rh-v1` is not.
PaperMinerToolkit stores whichever form the archive issued, unchanged, so the DOI in `chemrxiv_doi`
always resolves. Five suffix shapes are in use across the three platforms chemRxiv has run on, and
all of them are recognized. A `10.26434` DOI is never mistaken for a medRxiv or bioRxiv one.

chemrxiv.org is fronted by a bot challenge that can refuse a client outright. PaperMinerToolkit does not
try to get around it: a refusal is reported as the reason a search or download failed, and the same
papers stay reachable through the `openalex` and `crossref` sources.

`--count` is applied to each selected provider. Add `--store-abstract` to retain abstracts returned in search records immediately; otherwise abstracts can be fetched during downloading.

## Import local PDFs

Import every PDF in a directory:

```bash
pmt import pdfs papers/ papers.db
```

The importer scans PDF metadata and text for DOI candidates, looks up Crossref metadata, and stores the original PDF in the corpus. For offline imports, retain DOI extraction but skip Crossref:

```bash
pmt import pdfs papers/ papers.db --no-crossref
```

## Import an author's works

Crossref can seed a corpus before any content is downloaded. Prefer an ORCID:

```bash
pmt import author supervisor.db \
  --orcid 0000-0000-0000-0000 \
  --email you@example.ac.uk \
  --review-csv supervisor_works.csv
```

If no ORCID is available, use a full name and optionally an affiliation:

```bash
pmt import author supervisor.db \
  --author "First Family" \
  --affiliation "University of Example" \
  --email you@example.ac.uk
```

Inspect the review CSV before downloading. Crossref provides metadata and DOIs, not guaranteed access to paper content.

## Supplement metadata

Search and import records carry only a handful of bibliographic fields. `pmt enrich` fills in the rest
from Crossref, OpenAlex, PubMed, arXiv, medRxiv, bioRxiv, and chemRxiv:

```bash
pmt enrich papers.db
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
medRxiv and bioRxiv supply their subject category, under the `medrxiv_category` and
`biorxiv_category` schemes respectively, the licence the authors posted the preprint under, and
the link between a preprint DOI and the published DOI. They rank below arXiv for the same reason
and one more: a preprint row that names a published version already carries that version's DOI, so
Crossref and OpenAlex describe the paper better than the preprint record can. They are separate
sources rather than one because they are separate archives — a preprint DOI belongs to exactly one
of them, and asking the other costs a request and returns nothing.

chemRxiv supplies its subject categories under the `chemrxiv_category` scheme and the authors'
own keywords under `chemrxiv_keyword`, along with the licence and the link to a published version.
It files a preprint under more than one category where medRxiv and bioRxiv allow exactly one, so a
chemRxiv paper can carry several category rows, of which the first is flagged primary. It ranks
last for the same reason the other preprint servers rank low, and it supplies no full text.

Each provider's child rows are replaced independently, so enriching from one source never discards
another's.

Enrichment is re-runnable and resumable. A second run costs nothing because it only selects papers
that are still pending, and an interrupted run keeps everything it already committed:

```bash
pmt enrich papers.db --limit 500        # stop after 500 papers, resume later
pmt enrich papers.db --retry-failed     # retry only papers that previously failed
pmt enrich papers.db --refresh-after 90 # refresh citation counts older than 90 days
pmt enrich papers.db --force            # re-fetch everything
```

Restrict the providers, or skip reference lists when you only want the bibliographic fields:

```bash
pmt enrich papers.db --source crossref
pmt enrich papers.db --source openalex --no-references
pmt enrich papers.db --source pubmed
pmt enrich papers.db --source arxiv
pmt enrich papers.db --source medrxiv
pmt enrich papers.db --source biorxiv
pmt enrich papers.db --source chemrxiv
```

Papers with no DOI, OpenAlex identifier, PMID, arXiv identifier, medRxiv DOI, bioRxiv DOI, or
chemRxiv DOI are reported as skipped and cost no requests. A row that carries only one of those
identifiers — common for records found through PubMed, arXiv, medRxiv, bioRxiv, or chemRxiv
themselves — is resolvable whenever the provider that knows it is among the selected sources. Note
that none of the four preprint servers can be reached from an ordinary DOI: arXiv publishes no DOI
search field at all, and a published medRxiv, bioRxiv, or chemRxiv row carries the journal's DOI
rather than the preprint's. Each
enriches only rows that already carry the identifier it issued.
To supplement rows as they arrive instead of in a separate pass, add `--enrich` to discovery:

```bash
pmt search "Lithium solid electrolyte" papers.db --enrich
pmt import author supervisor.db --orcid 0000-0000-0000-0000 --enrich
```

`pmt reset` re-arms the enrichment stage without discarding enrichment data, so a reset corpus can be
re-enriched without refetching everything.

## Download abstracts, text, and PDFs

```bash
pmt download papers.db --format abstract
pmt download papers.db --format text
pmt download papers.db --format pdf
pmt download papers.db --format both
```

`--source` applies to abstracts, full text, and PDFs alike. Abstract retrieval tries OpenAlex,
PubMed, medRxiv, bioRxiv, chemRxiv, arXiv, CORE, and Elsevier in that order, skipping any source
the run did not select and any the row cannot reach; PubMed is attempted for any row carrying a
PMID or a DOI, and the preprint servers for any row carrying the identifier each issued. Full text
comes from Elsevier, PubMed Central, medRxiv, or bioRxiv, in that order and likewise only from the
selected sources -- so `--source elsevier --format text` uses Elsevier alone, and `--source pubmed`
no longer also reaches for it. PDF retrieval can use Unpaywall, OpenAlex, CORE, Elsevier, PubMed
Central, medRxiv, bioRxiv, chemRxiv, and arXiv, in that order. The four preprint
servers are tried last because the other sources may hold the publisher's version of record while a preprint server
holds the preprint, which is a different document. Select PDF sources by repeating `--source`:

```bash
pmt download papers.db --format pdf \
  --source unpaywall \
  --source openalex
```

PaperMinerToolkit tracks abstracts, full text, and PDFs independently. It skips a requested type when that type is already present while continuing to obtain missing types. Override this protection only when deliberately refreshing content:

```bash
pmt download papers.db --format both --force
```

Full text comes from Elsevier when a key is configured, and otherwise from the PubMed Central
open-access subset, which needs no credentials:

```bash
pmt download papers.db --format text --source pubmed
```

Only the PMC open-access subset is redistributable, so a paper with a PMC identifier outside that
subset reports that no full text is offered rather than failing. An NCBI API key raises the PubMed
request rate from three to ten per second, which matters most on large download runs; see
[Credentials and model configuration](configuration.md).

arXiv serves PDFs and abstracts but no full text, because it publishes no machine-readable
full-text format. Text for an arXiv paper comes from scraping its downloaded PDF.

medRxiv and bioRxiv are the other way round. Every posting on either names a JATS document, the
same format PubMed Central serves, so `--format text` takes their full text directly rather than
scraping a PDF:

```bash
pmt download papers.db --format text --source biorxiv
```

Their PDFs sit behind a bot challenge that occasionally refuses a client outright; when that
happens the run reports the refusal for that paper and carries on, and the text source is
unaffected.

chemRxiv is like arXiv rather than like the other two: it serves PDFs and abstracts but publishes
no machine-readable full text, so it is not a `--format text` source and `--source chemrxiv` is
rejected for that format. Text for a chemRxiv paper comes from scraping its downloaded PDF. Its
PDFs sit behind the same bot challenge, which PaperMinerToolkit reports rather than works around, so a
download run on a network the challenge refuses will report those papers and carry on.

By default, abstracts are also attempted while downloading text or PDFs. Disable that with
`--no-abstract`. Because `--source` now scopes abstracts too, a run narrowed to one provider
fetches abstracts from that provider alone; leave `--source` at its default to cast the wide net.

## Inspect the corpus

```bash
pmt corpus stats papers.db
pmt status papers.db
```

Corpus statistics include paper and asset counts, original and compressed storage sizes, and counts of text or abstract inputs that required chunking. Pipeline status summarizes search, download, scrape, and storage progress.

The database is the durable source of truth. CSV files produced by scraping, prediction, trends, and storage are exports or intermediate artifacts and can be recreated from the corpus and saved models.
