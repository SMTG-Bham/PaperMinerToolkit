# Credentials and model configuration

PaperScraper keeps search/download credentials separate from the text and vision model profiles. Secrets can be supplied through environment variables, which is best for batch jobs, or saved interactively with the `ps_*_key` commands.

## Search and download services

| Service | Environment variable | Interactive command | Used for |
| --- | --- | --- | --- |
| Elsevier | `ELSEVIER_API_KEY` | `ps_elsevier_key` | Scopus search, full text, and eligible PDFs |
| CORE | `CORE_API_KEY` | `ps_core_key` | Search, abstracts, and PDFs |
| OpenAlex | `OPENALEX_API_KEY` | `ps_openalex_key` | Higher API budget for search, abstracts, and OA locations |
| Unpaywall | `UNPAYWALL_EMAIL` | `ps_unpaywall_email` | Open-access PDF discovery |
| Crossref | `CROSSREF_EMAIL` | `ps_crossref_email` | Author imports and metadata enrichment |
| NCBI | `NCBI_API_KEY` | `ps_ncbi_key` | Higher PubMed and PMC request rate |
| NCBI | `NCBI_EMAIL` | `ps_ncbi_email` | Contact address sent to PubMed and PMC |

OpenAlex works without a key, but authenticated use has a substantially larger credit budget. The
OpenAlex `mailto` parameter identifies your client but no longer affects throughput, so
`ps_openalex_key` is the only setting that raises your budget.

Crossref has no API key. It asks automated clients to identify themselves with a contact address,
which `ps_crossref_email` stores once for `ps_import_author`, `ps_enrich`, and the Crossref lookup
that `ps_import_pdfs` performs. `--email` still overrides the stored value for a single command.

PubMed and PubMed Central need no credentials at all, but both NCBI settings are worth having.
NCBI paces unauthenticated clients at three requests per second and keyed clients at ten, counted
per IP address across every endpoint, so `ps_ncbi_key` is the single highest-leverage setting for
PubMed throughput: a 200-paper download run spends roughly three times less time waiting with a
key than without one. Keys are free from the Settings page of an NCBI account. `ps_ncbi_email`
stores the contact address NCBI uses to warn you before blocking an address; when it is unset,
PaperScraper reuses the Crossref address, so running `ps_crossref_email` covers both services.

arXiv needs no credentials and accepts no contact address, so there is nothing to configure
for it. arXiv asks that clients leave three seconds between consecutive requests, which
PaperScraper enforces itself; a search or enrichment run that spans many pages will spend a
noticeable part of its time waiting, and that is expected rather than a fault.

medRxiv needs no credentials either, and publishes no rate limit. PaperScraper paces itself at
one request a second regardless, which matters more here than for the other providers: medRxiv
has no search endpoint, so a search reads the posting archive a page at a time and a broad query
spends most of its run waiting between pages. Narrowing the query is what makes a medRxiv search
quick; see [Build a corpus](../user-guide/corpus.md).

## Hosted model providers

Save provider keys interactively or export `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`:

```bash
ps_openai_key
ps_anthropic_key
```

Configure text and vision independently:

```bash
ps_model_config text --provider openai --model YOUR_TEXT_MODEL
ps_model_config vision --provider openai --model YOUR_VISION_MODEL
ps_model_status
```

Use model identifiers available to your provider account. PaperScraper infers ordinary capabilities from the provider and model name; `--capability` is an override for unusual or locally served models.

## Local OpenAI-compatible servers

Point both profiles at the local endpoint when a model supports text and images:

```bash
ps_model_config text \
  --provider local \
  --model Qwen/Qwen3-VL-30B-A3B-Instruct \
  --base-url http://127.0.0.1:8000/v1

ps_model_config vision \
  --provider local \
  --model Qwen/Qwen3-VL-30B-A3B-Instruct \
  --base-url http://127.0.0.1:8000/v1
```

Requests default to `temperature=0` and `top_p=1` for repeatable extraction. Environment variables prefixed with `PAPERSCRAPER_MODEL_` configure the text profile; `PAPERSCRAPER_VISION_MODEL_` configures vision.

:::{warning}
Never put real keys in notebooks, recipe files, shell scripts, or committed configuration. Prefer your scheduler's secret mechanism or environment variables for unattended jobs.
:::
