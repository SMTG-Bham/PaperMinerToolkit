# Use the CLI from an AI agent

An agent can build and inspect a corpus by invoking `pmt` as a subprocess. Use
the installed command schema to discover arguments, request JSON on supported
commands, and retain the SQLite corpus between steps. This guide assumes the
agent has a working Python 3.11+ environment with PaperMinerToolkit installed.

## Discover the installed interface

```bash
pmt --version
pmt --help
pmt schema
pmt schema corpus export
pmt recipe list --json
pmt recipe show sse
pmt recipe prompt sse
```

`python -m paperminertoolkit` accepts the same arguments as `pmt`. This is useful
when an agent must use a particular virtual environment: invoke that
environment's Python with `-m paperminertoolkit`.

`schema` emits JSON without requiring `--json`. Its root contains
`schema_version` (currently `2`), `package_version`, `effects`, and `command`. Each command
describes its `name`, `path`, `help`, `parameters`, and nested `commands`.
The path is a string such as `pmt corpus list`.
Parameters include their names, argument/option type, required flag, defaults,
and type information, including choices where applicable. A command path such
as `corpus export` selects that subtree. Discover options from the installed
schema instead of assuming every command supports the same flags.

### Output schemas

Every command that can print JSON carries an `output_schema` on its schema
node: a JSON Schema of the result it prints. Commands without JSON output have
`null`. The schemas list the keys a result always carries as `required` and
allow additional keys, so new fields can appear without a schema version bump.
Nested provider and database records are described as plain objects. They use
only `type`, `properties`, `required`, `items`, `additionalProperties`, `enum`,
and `anyOf`. `scrape` has two shapes, the run summary and the `--dry-run` plan,
told apart by `dry_run`. The test suite checks every JSON result a command
prints against its schema, so the schema and the output cannot drift apart.

```bash
pmt schema corpus list | python -c "import json,sys; print(json.load(sys.stdin)['command']['output_schema'])"
```

### Side effects

Every runnable command declares what it can do to the world in an `effects`
list on its schema node, and the schema root carries the vocabulary as an
`effects` object of name and description. Check a command's effects before
running it. For example, select commands without `paid_model` while exploring,
or require explicit approval for `writes_corpus`.

| Effect | Meaning |
| --- | --- |
| `read_only` | Reads local files or the corpus without changing anything. |
| `migrates_corpus` | Opens the corpus normally, which upgrades an older corpus schema in place but changes no paper data. |
| `network` | Makes requests to external providers or model hubs. |
| `paid_model` | Calls the configured language model; hosted providers bill these requests. |
| `writes_corpus` | Changes papers, assets, statuses, or filters in the corpus database. |
| `writes_files` | Writes files outside the corpus, such as CSV, JSON, images, or model artifacts. |
| `writes_config` | Changes the saved PaperMinerToolkit settings file. |
| `prompts` | Can wait for input on stdin unless the documented options are supplied. |
| `starts_server` | Runs a local server until it is interrupted. |

Effects describe the default options. Some flags remove an effect: for
example, `--no-probe` avoids the network, and `--assume-yes` avoids the
`store` prompt. Groups such as `pmt corpus` declare no effects of their own.

Recipe listing, inspection, and prompt rendering do not call a model.
`recipe show` accepts a bundled name or a JSON recipe path and emits the loaded
recipe as JSON. `recipe prompt` emits plain text; `--outfile` saves that text.

## Configure an unattended run

Select providers explicitly. The default search source is `all`, and `--count`
limits results **per selected provider**, not across the whole search. A small
arXiv run needs no credentials. For other sources, inject only the needed
{doc}`provider credentials <configuration>` through the subprocess environment.

Start with local configuration checks, which make no network requests:

```bash
pmt probe --no-probe --source arxiv --json
pmt config status --json
```

Omitting `--no-probe` makes live provider requests. Some provider checks consume
credits; select the needed sources instead of probing every provider by default.
The `config status` JSON contains `text` and `vision` profiles with API-key
fields omitted.

For extraction, the runner can inject `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`
and configure a model through environment variables:

```bash
export PAPERMINERTOOLKIT_MODEL_PROVIDER=openai
export PAPERMINERTOOLKIT_MODEL_NAME=YOUR_TEXT_MODEL
pmt config status --json
```

Use a model identifier available to your account. The corresponding vision
variables are `PAPERMINERTOOLKIT_VISION_MODEL_PROVIDER` and
`PAPERMINERTOOLKIT_VISION_MODEL_NAME`. Local endpoints also use
`PAPERMINERTOOLKIT_MODEL_BASE_URL` or its vision equivalent. Alternatively,
`pmt config model text --provider openai --model YOUR_TEXT_MODEL` saves the
profile locally without prompting. To save provider credentials persistently,
read them from the environment rather than the command line:

```bash
pmt config openai-key --value-from-env OPENAI_API_KEY --json
pmt config crossref-email --value-from-env CROSSREF_EMAIL --json
printf '%s\n' "$CORE_KEY" | pmt config core-key --value-stdin --json
```

Keep secrets in the runner's environment, rather than command arguments,
logs, or recipe files.

The commands that need particular care in an unattended workflow are:

| Command | How to avoid a prompt |
| --- | --- |
| `config model` | Supply both `--provider` and `--model`; pass a key with `--api-key-from-env VAR`. Without a terminal, a missing option is a usage error rather than a prompt. |
| Credential setters such as `config openai-key`, `config crossref-email`, and `config core-rate` | Pass `--value-from-env VAR` or `--value-stdin`. Without either and without a terminal, they fail with a usage error instead of waiting. Keys checked online (OpenAI, Elsevier, OpenAlex) are still checked. |
| `store` | Supply `--assume-yes` after choosing the recipe and output file. This accepts conversions and skips unmatched columns. |

Close subprocess stdin so an unexpected prompt fails instead of waiting for a
person. Pass arguments as a list with `shell=False`, including each query and
path as a single argument. Use explicit corpus and output paths, and give each
run a known working directory.

## Acquire a bounded corpus

Create the output directory, then gather five abstracts from one provider:

```bash
mkdir -p run
pmt gather "lithium solid electrolyte" run/papers.db \
  --source arxiv --count 5 --format abstract --download-source arxiv --json
pmt corpus stats run/papers.db --json
pmt corpus searches run/papers.db --limit 5 --json
```

`gather` stores the search records and retrieves content only for papers matched
by that search. Its `search.paper_ids` are canonical corpus identifiers: retain
these opaque strings for later inspection. Repeating a query merges existing
records, and existing content is skipped unless `--force` is requested.

Discovery and downloading can also be separate:

```bash
pmt search "lithium solid electrolyte" run/papers.db \
  --source arxiv --count 5 --store-abstract --json
pmt download run/papers.db --format abstract --source arxiv --json
pmt enrich run/papers.db --source crossref --limit 5 --json
```

`download` considers the entire existing corpus. `gather` considers matched
papers and respects active filters. `search --store-abstract` stores abstracts present in search responses;
it does not guarantee that every paper has an abstract. Metadata enrichment
makes additional provider requests and is optional.

Text/PDF acquisition and non-abstract scraping include supporting information
by default. Select `--supplement-source` independently from article providers;
the repeatable choices are `all`, `structured`, `pubmed`, `figshare`, and
`publisher`. Use `--no-supplements` on acquisition or scraping to opt out.
Abstract-only runs skip SI. To add SI to existing entries without fetching
article content or abstracts:

```bash
pmt schema download
pmt download run/papers.db --format supplement --json
```

Retain all SI counts: `supplements`, `supplements_skipped`,
`supplements_failed`, `supplements_unavailable`, and
`supplement_discovery_failed`. File-download failures are counted separately
from papers with discovery errors; unavailable means completed discovery found
no SI. A missing supplement does
not imply a failed article download, and a successful exit code does not mean
every SI file was available. SI files retain separate source metadata and
per-asset text/image extraction checkpoints. Scraping can process newly added
SI even after article extraction succeeded. Consult {doc}`corpus` and
{doc}`scraping` for discovery coverage and supported formats.

## Inspect papers and export content

```bash
pmt corpus list run/papers.db --limit 5 --offset 0 --json
pmt corpus show run/papers.db "PAPER_ID_FROM_RESULTS" --json
pmt corpus export run/papers.db "PAPER_ID_FROM_RESULTS" run/abstract.txt \
  --role abstract --json
```

Replace `PAPER_ID_FROM_RESULTS` with an exact `paper_id` returned by search,
gather, or listing. Listing is deterministic and bounded: the default limit is
20, the maximum is 1000, and the offset defaults to zero. It returns `papers`,
`total`, `limit`, and `offset`. Advance the offset by the number of returned
papers until it reaches `total`; do not mutate the corpus while paging if the
same snapshot is required.

Add `--search-id SEARCH_ID` to restrict the listing to one recorded search.
Use `search_id` from a search result or `search.search_id` from a gather result;
an unknown search ID is an error. The returned `total` then counts that search's
matching papers.

`show` returns `paper` metadata and pipeline state, `assets` metadata for every
supplement and the latest article asset of each other role, and provenance in
`searches`, `authors`, `subjects`, and `references`. Binary content is excluded
from the JSON. Export selects a stored `abstract`, `text`, `pdf`, or `supplement`
asset and writes its content to the supplied file; its default role is `text`.
When several distinct SI files exist, select the exact `blob_id` shown in the
asset metadata:

```bash
pmt corpus export run/papers.db "PAPER_ID_FROM_RESULTS" run/supporting.pdf \
  --role supplement --asset-id "BLOB_ID_FROM_SHOW" --json
```

Existing output files require `--force`. A missing asset or ambiguous SI
selection is an error; export does not fetch content from a provider.

These three commands open the corpus read-only and require the current database
schema. They do not create or migrate a database. Missing, unsupported, or older
schemas fail with a diagnostic; use the normal corpus workflow with a compatible
package to perform any intended migration before inspection.

Older inspection commands such as `stats`, `searches`, `status`, and
`filter status` retain their normal database connection behavior, which can
migrate an older corpus when opening it.

## Extract, inspect progress, and resume

Once a text model is configured and the recipe has been reviewed, preview the
cost first. `scrape --dry-run --json` reports the papers, model requests, and
estimated input tokens without calling a model or changing the corpus:

```bash
pmt scrape run/papers.db sse --mode abstract --count 5 --dry-run --json
```

Then run it:

```bash
pmt recipe show sse
pmt scrape run/papers.db sse --mode abstract --count 5 \
  --output run/extracted.csv
pmt store run/papers.db run/extracted.csv run/materials.csv sse --assume-yes
pmt status run/papers.db --json
pmt filter status run/papers.db --json
```

`scrape` calls the configured model and writes an intermediate CSV. `store`
can also call the model for unit conversion, writes the final CSV, and updates
storage state. Both accept `--json`. `scrape` exits 0 when the run completes,
even if some papers failed, so read its `*_failed` counts and `failed_papers`
before moving on: a failure with `missing_source` needs `pmt download`, while
other failures are retried by a plain rerun. Keep the corpus, recipe, model
configuration, and both CSVs together.

### Preview changes with --dry-run

`reset`, `filter regex`, `filter topic`, `filter reset`, `store`, `download`,
and `scrape` accept `--dry-run`. A dry run reports what the command would
change and leaves the corpus and output files untouched, so an agent can check
the effect of a command that has `writes_corpus` before running it:

| Command | How the dry run works | What it reports |
| --- | --- | --- |
| `reset`, `filter regex`, `filter topic`, `filter reset` | Runs the real operation on a temporary copy of the corpus (made with SQLite's backup), then deletes the copy. | The same result as a real run, plus `dry_run: true`. Needs room for one corpus copy in `TMPDIR`. |
| `store` | Matches columns but converts nothing, so there is no model call, no prompt, and no file write. | `status: "dry_run"`, the columns it would convert or skip, and the rows it would write. |
| `download` | Applies the real skip rules to each selected paper without any request. | Which assets each paper would be asked for and which are already stored. Whether a provider would supply them is only known by asking. |
| `scrape` | See "Extract, inspect progress, and resume". | Papers, model requests, and estimated input tokens. |

The corpus must be at the current schema version. A dry run never migrates
it; it reports the mismatch instead.

### Follow long runs with --progress jsonl

`scrape`, `download`, `enrich`, `gather`, and `topics emergence` accept
`--progress auto|none|jsonl`.
`auto` (the default) shows tqdm progress bars, `none` hides them, and `jsonl`
hides them and writes one JSON event per line to stderr while the final result
stays on stdout:

```json
{"event": "run_start", "command": "scrape", "elapsed_s": 0.01, "total": 120, "mode": "text"}
{"event": "paper_done", "command": "scrape", "elapsed_s": 41.2, "paper_id": "arxiv:2401.0", "done": 1, "total": 120, "status": "failed", "materials": 0}
{"event": "run_end", "command": "scrape", "elapsed_s": 3600.5, "papers": 120, "materials": 311, "failed": 4}
```

Every event has `event`, `command`, and `elapsed_s`.

| Event | Emitted by | Fields |
| --- | --- | --- |
| `run_start` | `scrape`, `download` | `total` papers selected; `mode` or `format` |
| `paper_done` | `scrape` | `paper_id`, `done`, `total`, `status` (`succeeded`, `failed`, or `skipped`), `materials` |
| `paper_done` | `download` | `paper_id`, `done`, `total`, `status` (`fetched` or `nothing_new`), and `fetched` counts |
| `batch_done` | `enrich` | `papers` in the batch, `done`, `total` (the `--limit`, or null), and status `counts` |
| `batch_done` | `topics emergence` | `stage` (`embedding`), `papers` in the batch, `done` and `total` batches; papers already in the embedding cache are not re-embedded and report no events |
| `search_done` | `gather` (and `search` within it) | `search_id`, `status`, `result_count`, `papers_added`, `papers_updated` |
| `run_end` | `scrape`, `download`, `enrich` | The run's summary counts |
| `error` | any | `message` and `error_class` of the exception that ended the command |

Workflow messages also go to stderr, so read only the lines that start with
`{`: each is one complete event.

### Work with figures

`pmt figures` gets, inspects, and exports the figure images that image scraping
(`--image-extraction layout`) and `pmt digitize` use:

```bash
pmt figures download run/papers.db --json     # publisher figures named by stored XML
pmt figures detect run/papers.db --json       # figures detected from stored PDFs
pmt figures list run/papers.db --json         # what is stored, with scrape/digitize status
pmt figures export run/papers.db run/figures --json
pmt figures render paper.pdf run/render --strategy layout --json   # any PDF, no corpus
```

Each paper keeps figures from one origin, so a figure is never analysed twice.
`download` removes a paper's PDF-detected figures once its structured figures
arrive. `detect` skips papers that have structured figures, and papers already
detected unless `--force`, which replaces them. Like `scrape`, `download` and
`detect` follow active corpus filters unless `--ignore-filters` or `--paper-id`
is given. `render --strategy` covers the three image sources of `scrape`:
`layout`, `embedded`, and `pages`.

### Work on named papers

`scrape`, `download`, and `enrich` accept a repeatable `--paper-id` and a
`--paper-ids-file` (one ID per line; blank lines and `#` comments are ignored).
Use the `paper_ids` from `search`, `gather`, or `corpus list`, or the
`failed_papers` of a `scrape --json` summary, to test a recipe on one paper or
retry exactly the papers that failed:

```bash
pmt scrape run/papers.db sse --mode abstract --paper-id "PAPER_ID" --json
pmt download run/papers.db --format pdf --paper-ids-file run/retry.txt --json
```

An unknown ID is a usage error (exit 2) that names it, and nothing runs.
Active corpus filters do not apply to named papers. The usual skip rules still
do: a paper whose stage already succeeded is skipped unless `--force` is given,
and `enrich` also needs `--retry-failed` for failed papers. `scrape --order`
and `--count` still apply after the selection.

Inspect pipeline and filter status before scheduling more work.
`pmt scrape-progress run/papers.db text --json` answers what a rerun would
still do. It counts only the papers `scrape` would select, and lists the IDs of
failed papers (retried by a rerun) separately from those with no downloaded
document (which need `pmt download` first). Active filters
affect scraping and downloads within `gather`; standalone `download` considers
the entire corpus. Completed download and scrape stages are
skipped on a normal rerun; `--force` deliberately repeats completed work.
For enrichment, use `--retry-failed` to include failed papers and `--limit` to
bound each retry. Repeating `search` or `gather` adds a new history entry even
when records merge. `reset` changes pipeline state across the corpus and should
be reserved for deliberate reruns.

## Connect an MCP client

`pmt mcp` serves the CLI as Model Context Protocol tools over stdio, for
clients that call tools rather than run commands. Each JSON-capable command
becomes a tool: `pmt corpus list` is `corpus_list`, and `pmt scrape-progress`
is `scrape-progress`. Input schemas come from the command's parameters, output
schemas from `pmt schema`, and the declared effects become tool annotations
(`readOnlyHint`, `destructiveHint`, `openWorldHint`).

By default only read-only commands are tools. Admit more with `--allow`:

```bash
pmt mcp                                      # read-only tools
pmt mcp --allow network --allow writes_corpus --allow writes_files   # acquisition too
pmt mcp --allow paid_model ...               # scrape as well
pmt mcp --print-config                       # an mcpServers entry for this interpreter
```

Commands that change settings, prompt, or start a server (`config` setters,
`config model`, `store`, `validate gui`, `mcp`) are never exposed. Each tool
call runs `python -m paperminertoolkit <command> --json` as a subprocess with
stdin closed. The tool result is the command's JSON, or its error object with
`isError: true`. `--timeout` (default 3600 s) bounds each call.

The server implements the tool subset of MCP itself, with no extra dependency:
JSON-RPC 2.0 messages, one per line, with `initialize`, `ping`, `tools/list`,
`tools/call`, and `notifications/initialized`. It answers protocol versions
`2025-06-18`, `2025-03-26`, and `2024-11-05`, and sends output schemas and
`structuredContent` from `2025-06-18`. It has been tested against its own
offline session tests, not yet against every client.

## JSON and exit-code contract

JSON is **command-local**: write `pmt search ... --json`, not `pmt --json search`.
With `--json`, stdout always holds exactly one JSON value: the result, or an
error object when the command fails. Progress and the human-readable error
message go to stderr. `--help` remains human-readable even on commands that
support JSON.

### Error objects

A command run with `--json` that fails prints, on stdout:

```json
{"error": {"type": "not_found", "message": "Paper not found: arxiv:2401.0", "hint": null, "exit_code": 1}}
```

`exit_code` matches the process exit code. `hint`, when not null, suggests the
next step, such as the `pmt schema` command for a usage error. Configured
credentials are redacted from the message. `pmt schema` publishes the shape as
`error_schema` and the types as `error_types`:

| Type | Meaning |
| --- | --- |
| `usage` | Invalid command, option, or argument (exit 2); correct the invocation. |
| `not_found` | A named file, corpus, paper, asset, recipe, or model does not exist. |
| `io` | A file or the corpus database could not be read or written. |
| `provider` | An external provider or model service failed or could not be reached. |
| `operational` | The command ran but could not complete with these inputs or settings. |
| `internal` | An unexpected error; report it with the traceback on stderr. |

A command that has already printed its result does not add an error object.
For example, a partial search prints its summary and exits 1. Tell the cases
apart by the top-level `error` key. The error object follows any `--json` that
appears before a `--` separator. `schema` and `recipe show`, which print JSON
without the flag, report failures on stderr only unless `--json` is also given.

| Command | JSON result |
| --- | --- |
| `schema [COMMAND_PATH...]` | `schema_version`, `package_version`, and a `command` subtree; JSON by default. |
| `probe --json`, `config providers --json` | `providers` diagnostic rows and an `ok` boolean. |
| `search --json` | `search_id`, `status`, `sources`, `source_results`, `paper_ids`, `result_count`, `papers_added`, `papers_updated`, and `abstracts_stored`. |
| `gather --json` | `search` with the search result above, and `downloads` with acquisition counts or `null` if there were no matches. |
| `download --json` | `texts`, `pdfs`, `abstracts`, their `*_skipped` counts, and `supplements`, `supplements_skipped`, `supplements_failed`, `supplements_unavailable`, `supplement_discovery_failed`. |
| `enrich --json` | `succeeded`, `partial`, `not_found`, `unresolved`, `failed`, `authors`, `subjects`, `references`, and `batches` counts. |
| `corpus stats --json` | `stats` storage counts and `enrichment` metadata counts. |
| `corpus searches --json` | An array of recent search-history records, including provider outcomes. `--outfile` may also save the same JSON. |
| `corpus list --json` | `papers`, `total`, `limit`, and `offset`. |
| `corpus show --json` | `paper`, `assets`, `searches`, `authors`, `subjects`, and `references`. |
| `corpus export --json` | `paper_id`, `role`, `path`, and exported `bytes`; SI also includes `asset_id` and `original_filename`. |
| `digitize --json` | `requests`, `digitized`, `skipped`, `failed`, `not_found`, `supplementary`, `panels`, `series`, `points`, the `output` CSV path, and per-figure `errors`. |
| `scrape --json` | `papers_selected`, `papers`, per-stage `abstract_*`, `text_*`, and `image_*` counts (`attempted`, `skipped`, `succeeded`, `failed`), `missing_source`, `failed_papers` (`paper_id`, `stage`, `error`), `materials`, chunking counts (`chunked_inputs`, `chunk_requests`), `supplement_*` counts, `output_path`, `mode`, and `elapsed_seconds`. Token usage is not reported. |
| `scrape --dry-run --json` | `dry_run: true`, `model` (provider, name, `input_token_limit`, `usable_input_tokens`, `request_text_budget`), `tokenizer`, `tokenizer_fallbacks`, `prompt_tokens`, `papers_selected`, `papers_filtered_out`, `totals` (`papers_to_scrape`, `skipped_done`, `skipped_missing_source`, `requests`, `input_tokens`, `image_inputs`, `supplement_files_to_scrape`, `supplement_files_skipped`), `overflows`, and per-paper `papers` with stage and SI plans. No model is called and the corpus is not changed. |
| `scrape-progress --json` | `mode`, `scope`, `out_of_scope`, `filtered`, per-stage `stages` counts (`succeeded`, `no_asset`, `failed`, `not_attempted`), `supplements` file counts by status, `remaining` work a rerun would attempt, and per-stage `failed_papers` (retried by a rerun) and `no_asset_papers` (need `pmt download`) IDs. |
| `postprocess --json` | `records_in`, `records_out`, `audit_rows` (items the checks removed or edited, listed in the `.dropped.csv` file), `unchecked_papers` (no readable source text, so merged but not checked), `output_path`, and `dropped_path`. |
| `download --dry-run --json` | `dry_run: true`, `format`, `sources`, `supplement_sources`, `papers_considered`, `would_attempt` (`texts`, `pdfs`, `abstracts`, `supplement_discovery`), `already_stored` (`texts`, `pdfs`, `abstracts`, `supplement_files`), `no_text_source`, and per-paper `papers` with `attempt` and `stored` roles. |
| `store --json` | `status` (`stored`, `nothing_to_store`, `declined`, or `dry_run`), `dry_run`, `rows_read`, `rows_written`, `rows_total`, `converted_columns` (one unit-conversion model request each), `unconverted_columns`, `skipped_columns`, `papers_marked_stored`, `input_path`, and `output_path`. |
| `import pdfs --json` | `files`, `added`, `updated`, `doi_found`, `enriched`, `metadata_errors`, and the corpus `paper_ids` in file order. |
| `reset --json` | `papers_reset`, `supplement_stages_reset`, and the pipeline `columns` returned to their defaults; with `--dry-run`, also `dry_run: true`. |
| `status --json` | `papers`, `papers_with_structured_documents`, `papers_with_supplements`, `supplements`, `pipeline`, `text_materials`, and `image_materials`. |
| `filter status --json` | Active filter definitions and decision counts. |
| `import author --json` | `found`, `added`, `updated`, and `enriched` work counts, and the `review_csv` path. |
| `filter regex --json`, `filter topic --json`, `filter reset --json` | The resulting filter overview, in the same shape as `filter status --json`. |
| `topics train --json` | `model_dir` and a `report` with `documents_used`, `vocabulary_size`, `warnings`, and the training settings. |
| `topics compare --json` | `models_trained`, `output_dir`, `comparison_csv`, and per-model `models` diagnostics with their `warnings`. |
| `topics show --json` | `model_dir` and `topics`, each with `topic_id`, `topic_name`, `top_terms`, and `representative_papers` limited by `--representatives`. |
| `topics name --json` | `model_dir`, `topic_id`, and the stored `topic_name`. |
| `topics predict --json` | `papers_total`, `papers_predicted`, `papers_without_vocabulary_terms`, and `output_path`. |
| `topics trends --json` | `output_dir`, `trends_csv`, `report_json`, `windows`, `papers_missing_or_invalid_date`, and `plot_path` (or `null`). |
| `topics store --json` | `model_id`, `name`, `papers_predicted`, `papers_without_vocabulary_terms`, and `prediction_corpus_fingerprint`. |
| `topics models --json` | `models`, each with `name`, `model_id`, `is_current`, `num_topics`, `papers_predicted`, `papers_without_vocabulary_terms`, and `text_fields`. |
| `topics emergence --json` | The run summary also saved as `summary.json`: `output_dir`, `report`, `plots`, `papers_read`, `papers_used`, `dropped`, `end_year`, `settings`, the `areas`, `topics`, and `niches` counts, `topic_status`, `flagged_niches`, and `backtest_rho_score` (`null` without a backtest). |
| `entities corpus --json`, `entities text --json` | `entities` and `documents` counts and the JSONL `output` path; the corpus form also counts `papers`. |
| `figures download --json` | `papers`, `papers_without_structured_documents` (named papers that have none, else null), `papers_filtered_out`, `downloaded`, `skipped`, `failed`, `pdf_layout_replaced`, and `failures` (`paper_id`, `message`). |
| `figures detect --json` | `papers`, `papers_filtered_out`, `statuses` (`detected`, `none_found`, `skipped_structured`, `skipped_existing`, `no_pdf`, `failed`), `figures_stored`, `figures_replaced`, and `failures`. |
| `figures list --json` | `figures`, each with `paper_id`, `figure_id`, `label`, `caption`, `source`, `source_url`, `mime_type`, `bytes`, `blob_id`, `page_numbers`, `region_detected` (null for structured figures), `extraction_status`, and `digitization_status`; plus `total`, `limit`, and `offset`. |
| `figures export --json` | `outdir`, `count`, and `files` with `paper_id`, `figure_id`, `label`, `source`, `path`, and `bytes`. |
| `figures render --json` | `pdf`, `strategy`, `outdir`, `count`, `images` (`path`; for `layout` also `figure_id`, `label`, `caption`, `page_numbers`, `region_detected`), and `tables_detected` (null except for `layout`). |
| `validate score --json` | The full score: `fingerprint`, `sources`, `recipe`, `run`, `papers_evaluated`, `pending_pairs`, `missing_metadata`, `complete`, and per-field `fields` with `tp`, `fp`, `fn`, `pending`, `precision`, `recall`, `f1` (null when undefined), `type`, and `parts`. |
| `validate template --json` | `recipe`, the `outfile` path, and the header `columns` written. |
| `config model --json` | `profile`, `provider`, `model`, `base_url`, `capabilities`, `temperature`, `top_p`, `input_token_limit`, `reasoning_effort` and `thinking` (`null` for the model's default), and `api_key_set` (never the key). |
| `config elsevier-key --json`, `config core-key --json`, `config core-rate --json`, `config unpaywall-email --json`, `config crossref-email --json`, `config openalex-key --json`, `config ncbi-key --json`, `config ncbi-email --json`, `config openai-key --json`, `config anthropic-key --json` | `setting` (the settings key), `secret`, and `source` (`env`, `stdin`, or `prompt`); non-secret settings also report the saved `value`. Keys are never printed. |
| `config status --json` | `text` and `vision` model profiles, excluding API keys. |
| `recipe list --json` | An array of bundled recipe names. |
| `recipe show RECIPE` | The loaded recipe object; JSON by default. |
| `recipe optimise --json` | `best_recipe` and `history` paths, `baseline_score` and `best_score` (the selection F1), `papers`, and `holdout_papers`. Round-by-round progress goes to standard error. |

| Exit code | Meaning and next action |
| --- | --- |
| `0` | The command completed. Inspect result counts and per-paper state to determine whether the requested content or records are available. |
| `1` | An operational failure, failed provider probe, or incomplete search. Preserve any JSON and successful corpus writes; inspect stderr and provider outcomes before retrying. |
| `2` | A usage error such as an invalid option or argument. Correct the invocation; with `--json` the payload is a `usage` error object. |

A multi-provider search can emit a useful result with `status: "partial"` and
then exit with code 1. Successful papers remain in the corpus. A single-provider
failure exits before producing a summary, with an error object instead. Download/enrichment summaries can
report incomplete per-paper work while the command exits 0; main-article counts
do not include a total failure count, while SI has separate failed and
unavailable counts. Consult `status` and `corpus show` for those
details, rather than treating an exit code of 0 as proof every asset was fetched.
The `status.pipeline` object maps status-column names to observed state counts;
an absent state has count zero. For example, read failures with
`result["pipeline"]["text_download_status"].get("failed", 0)`.

The standard-library
[subprocess example](https://github.com/SMTG-Bham/PaperMinerToolkit/blob/main/examples/agent_cli.py)
uses the active Python environment, closes stdin, retains partial search JSON,
and prints a bounded listing of that search's matches. Running it without arguments or with
`--help` makes no provider requests:

```bash
python examples/agent_cli.py --help
python examples/agent_cli.py "lithium solid electrolyte" \
  --corpus run/papers.db --source arxiv --count 5
```

For a production runner, retain stderr alongside the JSON and apply its own
timeouts and retry budget. A timeout may leave completed corpus writes behind;
inspect the saved state before retrying.
