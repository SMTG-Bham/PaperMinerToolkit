# Credentials and model configuration

PaperMinerToolkit keeps search/download credentials separate from the text and vision model profiles. Secrets can be supplied through environment variables, which is best for batch jobs, or saved interactively with the commands under `pmt config`.

## Search and download services

| Service | Environment variable | Interactive command | Used for |
| --- | --- | --- | --- |
| Elsevier | `ELSEVIER_API_KEY` | `pmt config elsevier-key` | Scopus search, full text, and eligible PDFs |
| CORE | `CORE_API_KEY` | `pmt config core-key` | Search, abstracts, and PDFs |
| CORE rate | `CORE_MIN_INTERVAL` | `pmt config core-rate` | A faster CORE pace, if one was granted |
| OpenAlex | `OPENALEX_API_KEY` | `pmt config openalex-key` | Higher API budget for search, abstracts, and OA locations |
| Unpaywall | `UNPAYWALL_EMAIL` | `pmt config unpaywall-email` | Open-access PDF discovery |
| Crossref | `CROSSREF_EMAIL` | `pmt config crossref-email` | Author imports and metadata enrichment |
| NCBI | `NCBI_API_KEY` | `pmt config ncbi-key` | Higher PubMed and PMC request rate |
| NCBI | `NCBI_EMAIL` | `pmt config ncbi-email` | Contact address sent to PubMed and PMC |

OpenAlex works without a key, but a free one is worth ten times as much. Since February 2026
OpenAlex has metered a daily credit budget rather than a request rate: a client with no key gets
$0.10 of usage a day, and a free account with a key gets $1.00, both refilling at midnight UTC.
Those figures are the size of the free allowance priced in the units paid usage is billed in, not a
charge — the free tier needs no payment method, and exceeding it is refused rather than billed.
The `mailto` parameter identifies your client but no longer affects anything, so
`pmt config openalex-key` is the only setting that matters here, and like the Crossref address it
costs nothing.

Crossref has no API key. It asks automated clients to identify themselves with a contact address,
which `pmt config crossref-email` stores once for `pmt import author`, `pmt enrich`, and the Crossref lookup
that `pmt import pdfs` performs. `--email` still overrides the stored value for a single command.
Naming an address also doubles your rate: Crossref serves such clients from its polite pool at ten
requests per second rather than the public pool's five, at no cost. It is not required. A run without
one still works, at half the pace, and prints one line saying so. That makes this the cheapest
setting here — one command, no account, no key.

PubMed and PubMed Central need no credentials at all, but both NCBI settings are worth having.
NCBI paces unauthenticated clients at three requests per second and keyed clients at ten, counted
per IP address across every endpoint, so `pmt config ncbi-key` is the single highest-leverage setting for
PubMed throughput: a 200-paper download run spends roughly three times less time waiting with a
key than without one. Keys are free from the Settings page of an NCBI account. `pmt config ncbi-email`
stores the contact address NCBI uses to warn you before blocking an address; when it is unset,
PaperMinerToolkit reuses the Crossref address, so running `pmt config crossref-email` covers both services.

arXiv needs no credentials and accepts no contact address, so there is nothing to configure
for it. arXiv asks that clients leave three seconds between consecutive requests, which
PaperMinerToolkit enforces itself; a search or enrichment run that spans many pages will spend a
noticeable part of its time waiting, and that is expected rather than a fault.

medRxiv and bioRxiv need no credentials either, and their metadata APIs publish no rate limit.
They are one service under two names, so PaperMinerToolkit paces each API at one request per
second, which matters more here than for the other providers: neither has a search endpoint, so a
search reads the posting archive a page at a time and a broad query spends most of its run waiting
between pages. Narrowing the query is what makes either search quick, and it matters most for
bioRxiv, whose archive opened in 2013 and is several times the size of medRxiv's; see
[Build a corpus](corpus.md).

Their content sites are a different matter. `www.biorxiv.org` and `www.medrxiv.org`, which serve
the JATS full text, the PDFs, and the figures, each ask for seven seconds between requests in
their `robots.txt`, and are fronted by bot management that answers `429` with a `Retry-After` when
that pace is exceeded. PaperMinerToolkit honours the published delay, so downloading a preprint's
figures is deliberately slow: an eight-figure preprint spends about a minute. Requesting faster
does not help, because each refusal costs more waiting than the pace saved.

chemRxiv needs no credentials and publishes no rate limit either, and PaperMinerToolkit paces it at
one request per second as well. Unlike medRxiv and bioRxiv it does have a search endpoint, so a
broad query costs pages rather than a walk of the archive, and narrowing one saves the server
work rather than saving you a long read. chemrxiv.org is fronted by a bot challenge that can
refuse a client outright; PaperMinerToolkit does not try to get around it, and reports the refusal as
the reason a search or download failed. When that happens the same papers are still reachable
through the `openalex` and `crossref` sources.

## Request pacing

A rate limit belongs to the host being asked, not to the kind of file being fetched, so
PaperMinerToolkit paces each request by whichever host serves it. Several providers answer from one
host for metadata and another for content, which is why one provider can appear twice below with
two different delays.

| provider | metadata | text | PDF | figures | published limit |
|---|---|---|---|---|---|
| arXiv | 3.0 s | | 3.0 s | | ~1 request per 3 s requested |
| bioRxiv | 1.0 s | 7.0 s | 7.0 s | 7.0 s | API none; content `Crawl-delay: 7` |
| medRxiv | 1.0 s | 7.0 s | 7.0 s | 7.0 s | API none; content `Crawl-delay: 7` |
| chemRxiv | 1.0 s | | 1.0 s | | none published |
| CORE | 2.0 s | | 2.0 s | | 5 single or 1 batch request per 10 s |
| Crossref | 0.2 s | | | | 5/s public pool, 10/s with a contact address |
| Elsevier | 0.1 s | 0.1 s | 0.1 s | 0.1 s | 10/s article retrieval, 50k/week |
| OpenAlex | 0.01 s | 0.01 s | 0.01 s | | 100/s, then a daily credit budget |
| PubMed | 0.34 s | | | | 3/s, or 10/s with an API key |
| PMC Cloud Service | | 0.1 s | 0.1 s | 0.1 s | none published |
| Unpaywall | 0.1 s | | 0.1 s | | 100k/day |

A blank cell means the provider does not serve that kind of file: arXiv, chemRxiv, CORE, Crossref,
and Unpaywall publish no machine-readable structured document, so no figure reference of theirs is
ever downloaded, and Crossref serves metadata only.

Four consequences are worth knowing:

- **OpenAlex is limited by its budget, not by its pace.** The `0.01 s` is the hundred requests a
  second OpenAlex refuses above, but a run cannot sustain it for long: the daily credit budget runs
  out first, and a key raises the budget rather than the rate. PaperMinerToolkit reads the remaining
  credits from every response and refuses the next request once they are gone, naming when they
  refill, rather than letting a run discover it as a wall of refusals. OpenAlex answers both of its
  limits with `429`, so the two are told apart by the credits the response reports still available:
  some left means slow down and retry, none left means wait for midnight UTC.
- **A Crossref contact address halves the delay in the table.** The `0.2 s` shown is the public
  pool's pace, which is what an unconfigured client gets; `pmt config crossref-email` moves the run
  onto the polite pool at `0.1 s`, the same way an NCBI key moves PubMed from `0.34 s` to `0.11 s`.
  Crossref announces the allowance it is currently applying on every response, in
  `X-Rate-Limit-Limit` over an `X-Rate-Limit-Interval`, so either pace can be checked against the
  service rather than taken on trust.
- **Elsevier figures and Elsevier metadata share one budget.** Both come from `api.elsevier.com`,
  so they are paced by the same window and spend the same weekly quota. A large figure run reduces
  the article retrievals left for that week. Elsevier reports what is left of that quota on every
  authenticated response, and PaperMinerToolkit reads it: once nothing remains, the next request is
  refused before it is sent, naming the allowance and when it refills. That turns exhaustion into
  one clear error rather than a run of refusals that each cost a request. Nothing is enforced until
  a response has actually reported a figure, so an unmetered endpoint or an unauthenticated
  rejection never blocks a run.
- **PMC text, PDFs, and figures no longer touch NCBI's limit.** They come from the PMC Cloud
  Service, a separate service from E-utilities, so an NCBI API key does not change their pace and
  their traffic does not consume the E-utilities allowance.

CORE is the one provider whose pace is worth configuring. It publishes a single allowance for
unregistered clients — five single requests, or one batch request, per ten seconds, so two seconds
apart and ten respectively — and grants faster paces individually to registered organisations,
which Supporting and Sustaining members receive as a membership benefit. CORE publishes no figure
for any of those faster paces, not even in its membership documentation, which describes what kind
of access each level brings rather than at what rate. So if CORE granted you a pace, set it:

```bash
pmt config core-rate
```

The prompt states the free allowance it replaces, and leaving it blank returns to that allowance.
Batch methods stay five times slower than single ones, mirroring CORE's own two allowances.

Where a provider answers `429` with a `Retry-After`, PaperMinerToolkit waits for the interval the
service asks for rather than its own backoff curve, and retries. A refusal is therefore usually
survivable rather than fatal, but it is slower than pacing correctly in the first place.

## Hosted model providers

Save provider keys interactively or export `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`:

```bash
pmt config openai-key
pmt config anthropic-key
```

Configure text and vision independently:

```bash
pmt config model text --provider openai --model YOUR_TEXT_MODEL
pmt config model vision --provider openai --model YOUR_VISION_MODEL
pmt config status
```

Use model identifiers available to your provider account. PaperMinerToolkit infers ordinary capabilities from the provider and model name; `--capability` is an override for unusual or locally served models.

## Local OpenAI-compatible servers

Point both profiles at the local endpoint when a model supports text and images:

```bash
pmt config model text \
  --provider local \
  --model Qwen/Qwen3-VL-30B-A3B-Instruct \
  --base-url http://127.0.0.1:8000/v1

pmt config model vision \
  --provider local \
  --model Qwen/Qwen3-VL-30B-A3B-Instruct \
  --base-url http://127.0.0.1:8000/v1
```

Requests default to `temperature=0` and `top_p=1` for repeatable extraction. Environment variables prefixed with `PAPERMINERTOOLKIT_MODEL_` configure the text profile; `PAPERMINERTOOLKIT_VISION_MODEL_` configures vision.

:::{warning}
Never put real keys in notebooks, recipe files, shell scripts, or committed configuration. Prefer your scheduler's secret mechanism or environment variables for unattended jobs.
:::
