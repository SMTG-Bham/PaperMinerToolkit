# A full campaign, end to end

The {doc}`quickstart` scrapes five abstracts. This page follows a production run instead: a
corpus of about 1,300 papers built from several queries, filtered on full text, scraped with a
locally served model on a cluster, checked, and rescraped where it failed, then stored and handed
to figure digitising. The numbers come from a polymer-degradation campaign run with the bundled
`polymer_degradation` recipe on Intel Gaudi nodes. That campaign had been gathered, filtered,
scraped, rescraped and reviewed when this page was written; the storing step shows the commands
only, and the digitising step reports a five-figure trial. The commands are the general ones, and
the same sequence applies to any recipe.

Each stage is resumable and skips what is already stored, so every command below can be
rerun after an interruption without repeating finished work.

## 1. Plan the queries and the recipe

Write one query per line in a text file. Providers differ in what they search (Scopus wraps a
query in `TITLE-ABS-KEY()`, OpenAlex uses its `search` parameter, CORE its `q` parameter), but
`AND`, `OR` and quoted phrases work across OpenAlex, CORE and PubMed:

```text
polymer AND ("mass loss" OR "weight loss") AND (biodegradation OR "soil burial" OR composting)
polymer AND ("mass loss" OR "weight loss") AND ("hydrolytic degradation" OR photodegradation)
(plastic OR polymer) AND (biodegradation OR mineralization) AND ("ISO 14855" OR "ASTM D5338")
```

Several narrow queries find more relevant papers than one broad one, and records that several
queries return are merged on DOI rather than duplicated.

Inspect the recipe before spending model time on it, and check that its system prompt and a
typical reply fit the model's context:

```bash
pmt recipe show polymer_degradation
pmt recipe prompt polymer_degradation
```

See {doc}`recipes` for writing your own, including the `values` vocabularies that fold a model's
synonyms onto fixed categories.

## 2. Gather the corpus

Run `pmt gather` once per query and search provider. Searching and downloading are network
work, so run them as ordinary CPU batch jobs, not on an accelerator:

```bash
pmt gather "polymer AND (\"mass loss\" OR \"weight loss\") AND composting" papers.db \
  --source openalex --count 200 --format both \
  --download-source openalex --download-source unpaywall --download-source core \
  --supplement-source pubmed --supplement-source structured \
  --json > gather_q1_openalex.json
```

`--format both` asks for full text and PDFs; abstracts come too unless `--no-abstract` is given.
`--download-source` and `--supplement-source` restrict where content and supporting information
are fetched from. Leave out a provider that cannot serve you: in this campaign Elsevier was left
out because the key had no full-text entitlement from the cluster's network, and Figshare and
publisher SI pages refused automated requests.

A wrapper that loops over the query file and records a marker per finished (query, provider)
step lets a job killed at its wall-clock limit be resubmitted without repeating earlier steps.
The `--json` summary of each step records how many papers were added and downloaded.

Two things to expect at this scale:

- **Rate budgets run out.** OpenAlex's daily credit budget was spent partway through this
  gather, and 342 papers recorded that error. A rerun of `pmt download` after the budget resets
  retries them; most turned out to be paywalled anyway.
- **Not every paper gets full text.** Of 1,288 papers, 465 had text and 490 had PDFs.

```bash
pmt corpus stats papers.db
pmt status papers.db
```

## 3. Filter on full text

A search matches words in titles and abstracts; a regex filter can demand evidence in the full
text. The campaign's filter, `examples/filters/polymer_degradation_text.json`, requires both a
degradation term and a measured outcome such as mass loss or mineralisation:

```bash
pmt filter regex papers.db examples/filters/polymer_degradation_text.json --dry-run
pmt filter regex papers.db examples/filters/polymer_degradation_text.json --replace
pmt filter status papers.db
```

Here 849 papers were included, 37 excluded, and 402 had no full text to judge. Scraping and
later stages follow the active filter. See {doc}`filtering` for combining regex and topic filters.

## 4. Preview, then scrape

Check what a scrape would send before paying for it. `--dry-run` counts requests and tokens with
the configured model's tokenizer and flags papers that will need chunking, without calling the
model:

```bash
pmt scrape papers.db polymer_degradation --mode text --dry-run --json
```

Then scrape one paper as a smoke test, and the whole corpus once it looks right:

```bash
pmt scrape papers.db polymer_degradation --mode text --count 1 --output scraped.csv
pmt scrape papers.db polymer_degradation --mode text --output scraped.csv --progress jsonl
```

On a cluster, run the scrape inside the same allocation that serves the model (see
{doc}`hpc`), and resubmit the same command to continue: completed papers are skipped. Do not use `--count` to pace a long run, because it slices the ordered work list.

Plan the wall time from a measured rate, not a guess. With Qwen3-30B-A3B serving four cards,
this recipe took about 95 to 180 s per paper, because its replies run to many records and
generation, not reading, dominates. 849 papers took three 12-hour allocations, the first cut
short after 9 hours by a node failure; the next one resumed where it stopped.

## 5. Check progress and retry what failed

`pmt scrape-progress` reports what a rerun would still do, separating papers that failed (a
rerun retries them) from papers that have no document (they need `pmt download` first):

```bash
pmt scrape-progress papers.db text --json
```

After the first pass, 457 papers had succeeded, 391 had no downloaded document, one PDF was
corrupt, and 45 supporting-information files had failed because they were spreadsheets too long
for the model's context. Fixing the cause and rerunning the same scrape retried only those 45
files and the one paper; 198 SI files then succeeded and 6 remained.

## 6. Review the records before storing them

Read the raw CSV before `pmt store`, because storing converts units in place and consumes the
input file. Keep a copy first. Things worth checking, with what this campaign found:

- **Blank key fields.** 366 rows had no degradation pathway; nearly all also lacked a test type
  and a mass loss, which marks them as background measurements rather than degradation records.
- **Copied prompt examples.** A few rows repeated the recipe's example sentences word for word.
  Placeholder examples and the `Value warnings` column now catch most of these; see
  {doc}`recipes`.
- **Values outside a field's vocabulary,** such as `enzymatic` for a pathway field whose values
  are `biotic` or `abiotic`. Declaring `values` and `value aliases` in the recipe folds them.
- **Single papers that dominate.** Four microbiome papers contributed 632 of the 724 rows from
  the SI rescrape.

{doc}`validation` describes scoring a scrape against a hand-checked reference with
`pmt validate score`, which turns these spot checks into numbers.

## 7. Store

```bash
cp scraped.csv scraped_raw.csv
pmt store papers.db scraped.csv materials.csv polymer_degradation --dry-run --json
pmt store papers.db scraped.csv materials.csv polymer_degradation --assume-yes
```

The dry run lists which columns would be converted (one model request each) and which would be
skipped, without changing anything. A real run consumes `scraped.csv`, which is why the copy is
taken first.

## 8. Recover curves from the figures records cite

The recipe's `Mass loss figure` field names the figure that plots each record's mass loss. 491
records named a figure but gave no values, and {doc}`digitizing` recovers those curves. Run it on
the raw copy of the scraped CSV, whose column names are the recipe's field names; stored headings
carry unit labels:

```bash
pmt figures detect papers.db
pmt digitize papers.db scraped_raw.csv \
  --figure-field "Mass loss figure" --only-missing "Mass loss curve" \
  --output mass_loss_curves.csv --previews previews --limit 5
```

Try a handful first. A five-figure trial on this corpus recovered none: four cited figures were bar charts of
mass loss per sample rather than curves, which plottracer does not read; they suit an image
scrape with a vision model instead. `pmt figures list papers.db --json` shows which figures were
detected and their digitising status.

## What to keep

Keep the corpus, the query file, the recipe (or its bundled name and the package version), the
filter rules, the raw and stored CSVs, the model configuration and the scheduler logs. Together
they reproduce the dataset, and the corpus records which provider supplied every document.
