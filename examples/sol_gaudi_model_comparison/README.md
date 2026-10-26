# Comparing extraction models on ASU Sol's Gaudi nodes

This example scrapes one corpus with several language models and reports how
the extracted records depend on the model. It uses the bundled
`band_gap_validation` recipe and a small/medium/large lineup of Qwen
instruction models, each served by vLLM on Intel Gaudi 2 cards from a Slurm
job. Everything is driven by three scripts plus a lineup file; the toolkit
itself is unchanged. Read [`../sol_gaudi/README.md`](../sol_gaudi/README.md)
first: it explains the two-environment layout (ASU's shared `gaudi-pytorch-vllm`
for the server, your own `paperminertoolkit` for the client), the context and
memory knobs, and the troubleshooting notes. This example reuses its installer
and preflight check.

| File | Runs on | Purpose |
| --- | --- | --- |
| [`models.tsv`](models.tsv) | — | The lineup: label, Hugging Face repo id, Gaudi cards, size class |
| [`prefetch_weights.sbatch`](prefetch_weights.sbatch) | `htc` | Download the weights to scratch, off the Gaudi queue |
| [`scrape_model.sbatch`](scrape_model.sbatch) | `gaudi` | Serve one model and scrape one copy of the corpus with it |
| [`submit_comparison.sh`](submit_comparison.sh) | login node | Submit one `scrape_model.sbatch` per lineup entry |
| [`summarize_comparison.py`](summarize_comparison.py) | login node | Tabulate coverage, records, fill rates, throughput and model agreement |

Everything is submitted from this directory and written back under
`runs/<label>/`, which is ignored by git.

## The lineup

| Label | Model | Cards | Class | Why |
| --- | --- | --- | --- | --- |
| `qwen25-7b` | `Qwen/Qwen2.5-7B-Instruct` | 1 | small | Cheapest to queue; the floor |
| `qwen25-32b` | `Qwen/Qwen2.5-32B-Instruct` | 2 | medium | Mid-sized dense model |
| `qwen3-30b-a3b` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | 4 | medium-moe | Mixture of experts with 3B active parameters; the default in `../sol_gaudi`, and on Intel's Gaudi 2 validated list |
| `qwen25-72b` | `Qwen/Qwen2.5-72B-Instruct` | 4 | large | Largest ungated dense model that fits four cards with a 32768-token context |

Keeping to one family isolates the effect of size from the effect of training
data and chat template. All four are ungated, so no Hugging Face token is
needed. To add a gated model such as `meta-llama/Llama-3.1-70B-Instruct` (TP 4),
accept its licence on Hugging Face, put `export HF_TOKEN=...` in
`~/.paperminertoolkit_env` (mode 600), and add a line to `models.tsv`. Avoid
`-Thinking-` variants: they spend the fixed 10000-token completion budget on
reasoning before writing any JSON.

The card counts come from the weights: BF16 takes 2 bytes per parameter and a
Gaudi 2 card has 96 GB, so 72B needs at least two cards and four leaves room
for the KV cache. Every job serves at `--max-model-len 32768`, which all four
models support without RoPE scaling.

## 1. Install

The installer in the neighbouring example builds the `paperminertoolkit`
environment, keeps CUDA torch out of it, and runs the test suite. Submit it
once, from its own directory, and wait for it to finish:

```bash
cd examples/sol_gaudi
sbatch install.sbatch
tail -f ps-install-<jobid>.log
```

## 2. Prefetch the weights

Transfers belong on `htc`, not in a Gaudi allocation. One array task per
lineup entry:

```bash
cd examples/sol_gaudi_model_comparison
sbatch --array=0-3 prefetch_weights.sbatch
```

The 72B model is about 145 GB; the four together are about 290 GB under
`$HF_HOME` (`/scratch/$USER/hf` by default). Scratch has no backups and a
90-day limit, so re-run this step before a later campaign; it only fetches
what is missing.

## 3. Build or choose a corpus

Any PaperMinerToolkit corpus works. For a new one, use
[`../sol_gaudi/fetch_corpus.sh`](../sol_gaudi/fetch_corpus.sh) with a band-gap
query, for example:

```bash
../sol_gaudi/fetch_corpus.sh "band gap semiconductor DFT optical absorption" /scratch/$USER/corpus/papers.db 200
```

The scrape runs in `text` mode, which uses the stored full text or PDF of each
paper. Papers with only an abstract fail with "No downloaded text or PDF asset
found" and are counted as failed, identically for every model. Check coverage
before spending Gaudi time:

```bash
pmt corpus stats /scratch/$USER/corpus/papers.db
```

The source corpus is never written to. On its first job each label copies it
to `runs/<label>/papers.db` with SQLite's backup API, because scrape statuses
live in the corpus and two models sharing one database would skip each other's
papers.

## 4. Smoke test, then the full run

Preview what would be submitted:

```bash
./submit_comparison.sh --corpus /scratch/$USER/corpus/papers.db --count 5 --dry-run
```

Then submit five papers per model. Every job orders the corpus by paper id,
so the five papers are the same for all four models:

```bash
./submit_comparison.sh --corpus /scratch/$USER/corpus/papers.db --count 5
squeue -u $USER -n bgcmp-qwen25-7b,bgcmp-qwen25-32b,bgcmp-qwen3-30b-a3b,bgcmp-qwen25-72b
tail -f runs/qwen25-7b/scrape-<jobid>.log
```

Each job gets its own card count from `models.tsv`. One-card jobs usually start
first. When the logs end with `=== Done ===` and the records look sane, run the
whole corpus:

```bash
./submit_comparison.sh --corpus /scratch/$USER/corpus/papers.db
```

Do not reuse `--count` to continue a partial run: it reselects the same first
N papers. Jobs default to 12 hours (`--time` changes it; the `gaudi` partition
allows up to 7 days). The scrape is resumable, so if a job hits its wall clock
resubmit only that label and it continues from the last committed paper:

```bash
./submit_comparison.sh --corpus /scratch/$USER/corpus/papers.db --only qwen25-72b
```

Other knobs pass through `--export`, exactly as in `scrape_gaudi.sbatch`, for
example `sbatch --export=ALL,PMT_MAX_MODEL_LEN=65536,... scrape_model.sbatch`;
run `./submit_comparison.sh --dry-run` to see the full command for a label and
edit it.

### What each run records

`runs/<label>/run_history.jsonl` gets one line per job with the model, its
Hugging Face revision, tensor-parallel size, context settings, recipe, mode,
order, the vLLM and PaperMinerToolkit versions, the repository commit, job id,
node, start and end times and exit status. Together with the greedy decoding
the toolkit uses by default (`temperature=0`), that is what a reader needs to
rerun one model.

## 5. Summarise

Once every job has finished:

```bash
python summarize_comparison.py
```

It prints three Markdown tables and writes `runs/comparison_summary.csv` and
`runs/comparison_agreement.csv`:

- **Coverage and throughput**: papers succeeded, failed and untouched per
  model, total wall hours across resubmissions, papers per hour, and the most
  common recorded error.
- **Records**: record count, papers with at least one record, records per
  succeeded paper, the share of records with a populated `Band gap`,
  `All band gaps` and `Cited literature band gaps`, items per record, and how
  many list cells were not valid JSON.
- **Pairwise agreement**: for every pair of models, over the papers both
  scraped successfully, the mean Jaccard similarity of the material names
  (normalised to lower-case alphanumerics) and of the numeric principal
  band-gap values (two decimals). Two models that both return nothing for a
  paper count as agreeing.

Agreement is not accuracy. A small model can agree with a large one on a
wrong value, and the large one can be right alone. The tables tell you where
models diverge, which is where to spend review time.

## 6. Validate against a curated reference

`pmt validate` scores each model's CSV against the same human-curated
reference, which is the comparison that matters. Make the template once,
fill it for a subset of papers (the ones the agreement table flags are a good
start), then review each model's CSV:

```bash
pmt validate template band_gap_validation band_gap_validation.csv
# ... curate band_gap_validation.csv ...
pmt validate gui --validation band_gap_validation.csv \
  --scraped runs/qwen25-7b/scraped_band_gap_validation.csv \
  --recipe band_gap_validation --no-browser
```

The reviewer needs a browser; forward its port from Sol or copy the two CSVs
to a workstation. Fill in the model identifier, context length, commit and
run date from `run_history.jsonl` in the scoring summary, and download one
scoring run per model. See
[the validation guide](https://paperminertoolkit.readthedocs.io/en/latest/workflow/validation.html)
for the review rules and metric definitions.

## Notes

- `pmt store` is not run. The band-gap recipe has no unit-bearing fields, so
  storing would make no model calls; the scraped CSV is what `pmt validate`
  reads.
- Supporting information is included by default (`PMT_SUPPLEMENTS=0` turns it
  off), so papers whose gap tables sit in the SI are extracted the same way
  for every model.
- Only `Qwen3-30B-A3B-Instruct-2507` is on Intel's Gaudi 2 validated list.
  The dense Qwen2.5 models are validated on Gaudi 3 and run on Gaudi 2 through
  the same code path; if one fails during HPU graph warmup, lower
  `PMT_GPU_MEMORY_UTILIZATION` or `PMT_MAX_NUM_SEQS` for that label first.
- Newer families (Qwen3.5, Qwen3.8, Gemma 4) need a vLLM Gaudi plugin newer
  than the one in ASU's shared environment and are not runnable here yet.
