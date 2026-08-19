# Running on ASU Sol's Intel Gaudi nodes

This guide gets PaperScraper running against a local vLLM server on Sol's Gaudi 2
accelerators. It replaces [`examples/qwen_vllm_workflow.ipynb`](../../examples/qwen_vllm_workflow.ipynb),
which assumes a CUDA workstation.

The worked example is biodegradable polymers: the scripts default to the
`polymer` recipe and an OECD 301 search query. Everything here applies to any
recipe — override `PS_RECIPE` and the query if you are after something else.

The important thing to understand first: **PaperScraper does no accelerator work
of its own.** It is an HTTP client that speaks the OpenAI chat API. The Gaudi
cards are used only by a separate vLLM server process. So this is not a code
port — it is two environments and an sbatch script.

Those two environments cannot be merged. The vLLM Gaudi stack is built on Python
3.10; PaperScraper requires 3.11 or newer. They talk over `localhost` instead.

| Environment | Owner | Python | Contents |
| --- | --- | --- | --- |
| `gaudi-pytorch-vllm` | ASU Research Computing (shared) | 3.10 | vLLM + Habana PyTorch |
| `paperscraper` | you | 3.14 | PaperScraper and its dependencies |

> **Never `pip install` into `gaudi-pytorch-vllm`.** It is shared by everyone on
> Sol, and pip would happily replace Habana's patched PyTorch with a stock wheel,
> silently breaking HPU support for every other user. Only ever `source activate`
> it.

The runnable scripts live in [`examples/sol_gaudi/`](../../examples/sol_gaudi/),
alongside the workflow notebooks:

| Script | Runs on | Purpose |
| --- | --- | --- |
| [`install.sbatch`](../../examples/sol_gaudi/install.sbatch) | `htc` | **Run this first.** Builds the environment and verifies it |
| [`fetch_corpus.sh`](../../examples/sol_gaudi/fetch_corpus.sh) | login node or interactive | Search and download; never touches the model |
| [`fetch_corpus.sbatch`](../../examples/sol_gaudi/fetch_corpus.sbatch) | `htc` | The same thing as a batch job, for corpora too big to babysit |
| [`scrape_gaudi.sbatch`](../../examples/sol_gaudi/scrape_gaudi.sbatch) | `gaudi` | **Start here.** vLLM + scrape + store in one job |
| [`serve_gaudi.sbatch`](../../examples/sol_gaudi/serve_gaudi.sbatch) | `gaudi` | Long-lived server for reusing a warm model |

Copy them into your working directory and edit, or submit them in place — every
setting is an environment variable with a default.

## Quick start

From the repository root, three submissions end to end:

```bash
sbatch examples/sol_gaudi/install.sbatch
sbatch examples/sol_gaudi/fetch_corpus.sbatch
sbatch examples/sol_gaudi/scrape_gaudi.sbatch
```

Wait for each to finish before the next — the corpus has to exist before the
scrape, and the environment before either. Watch progress with
`tail -f ps-install-<jobid>.log`.

Sections 1 to 4 explain what `install.sbatch` does and how to do it by hand;
skip to section 5 if the install job succeeded.

## 1. Get an interactive session

Only needed if you are installing by hand. ASU asks that environments not be
built on a login node, which is why `install.sbatch` is a batch job.

```bash
interactive -c 4 -t 0-2
```

That is shorthand for `salloc -c 1 -p htc -q public -t 0-4` with your overrides.

## 2. Create the PaperScraper environment

```bash
module load mamba/latest
mamba env create -f build_tools/environment.yml
source activate paperscraper
```

[`build_tools/environment.yml`](../environment.yml) supplies only the interpreter;
every dependency comes from `pyproject.toml`. Use `source activate`, never
`mamba activate` — the latter writes cruft into your shell configuration.

## 3. Install the package, without dragging in CUDA

ASU's [vLLM page](https://docs.rc.asu.edu/vllm/) tells you to use the `mamba`
module or an apptainer image, and — verbatim — "Do not use `uv`, `conda`, or
`pip`, or docker." That rule is about building your own inference stack, and the
PyTorch-mismatch entry under Troubleshooting is exactly what it exists to
prevent. What follows stays inside it: the environment comes from the `mamba`
module, torch and vLLM are only ever the shared `gaudi-pytorch-vllm` copies, and
`pip` installs nothing but PaperScraper and its pure-Python dependencies into an
environment of our own — never the shared one, never `~/.local`. If your group
reads the rule more strictly than that, build a `.sif` and run the scraper from
it instead; nothing in PaperScraper needs the host environment.

`pyproject.toml` depends on `headroom-ai[image,ml]`, whose `ml` extra requires
`torch`. Left alone, pip resolves that to the CUDA build and pulls several GB of
`nvidia-*` wheels and `triton` that are useless on a Gaudi node. PaperScraper only
touches torch indirectly, through a lazy `transformers` import used for token
counting, so the CPU build satisfies it completely.

Install CPU torch first so pip's constraint is already met:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[test]"
```

Verify:

```bash
pytest
python -c "import torch; print(torch.__version__)"
```

`pytest` needs no credentials and no accelerator — `pyproject.toml` already
excludes the `network` and `slow` markers by default. The torch version should
carry a `+cpu` suffix.

`install.sbatch` checks all of this and fails the job if any of it is wrong: a
torch build without `+cpu`, any surviving `nvidia-*` or `triton` package, a
missing `ps_*` console script, or a failing test. It neither reads nor writes
`~/.local` (`PYTHONNOUSERSITE=1`, `PIP_USER=0`), since a wheel landing there
shadows both this environment and the shared Gaudi one — see Troubleshooting. It
also pre-caches the tokenizer for `PS_MODEL`, so the Gaudi job does not need the
network for the chunk-sizing lookup. Pass `PS_PREFETCH_WEIGHTS=1` to download
the model weights too, which keeps tens of GB of transfer out of your four-hour
accelerator allocation:

```bash
sbatch --export=ALL,PS_PREFETCH_WEIGHTS=1 examples/sol_gaudi/install.sbatch
```

## 4. Point caches at scratch

`/home` is capped at 100 GiB and model weights run 15–140 GB, so keep the
Hugging Face cache on scratch. Add to your `~/.bashrc`:

```bash
export HF_HOME="/scratch/$USER/hf"
export PIP_CACHE_DIR="/scratch/$USER/pip"
```

This matters on both sides. vLLM downloads the weights, and PaperScraper loads
the *tokenizer* for the same model name to size its text chunks — sharing one
`HF_HOME` means the second lookup is a cache hit rather than a re-download.

## 5. Set credentials

Searching and downloading need publisher credentials; the local model does not
need an OpenAI or Anthropic key.

```bash
export ELSEVIER_API_KEY="..."
export CORE_API_KEY="..."
export UNPAYWALL_EMAIL="you@asu.edu"
```

The interactive `ps_elsevier_key`, `ps_core_key`, and `ps_unpaywall_email`
commands store these in `~/.config/.pscraperrc.json` if you prefer.

## 6. Build the corpus off the Gaudi queue

Search and download are network-bound and never call the model, so they must not
burn a Gaudi allocation.

```bash
./examples/sol_gaudi/fetch_corpus.sh
```

That uses the default query, `biodegradable polymer OECD 301 biodegradation`.
Pass your own as the first argument:

```bash
./examples/sol_gaudi/fetch_corpus.sh "polymer biodegradation OECD 310" papers.db 100
```

Naming the test standard — OECD 301, OECD 310, ASTM D6400, ISO 14855 — tends to
surface papers with extractable results rather than reviews, which matters
because the biodegradation fields are only filled when a paper states them.

For a corpus large enough that you do not want to sit in an interactive session
waiting on rate limits, submit it instead:

```bash
sbatch examples/sol_gaudi/fetch_corpus.sbatch
sbatch --export=ALL,PS_COUNT=500 examples/sol_gaudi/fetch_corpus.sbatch
```

That runs on `htc` with no accelerator requested, since neither stage calls the
model. `htc` caps wall time at four hours but runs uninterrupted; for a bigger
corpus switch the header to `-p general`, which allows up to a week on the
`public` QOS. Submit from the repository root so `papers.db` lands there rather
than in the scheduler's spool directory.

A batch job does not reliably inherit your shell environment, so put the
credentials in a file only you can read and the script will source it:

```bash
printf 'export ELSEVIER_API_KEY=...\nexport CORE_API_KEY=...\nexport UNPAYWALL_EMAIL=you@asu.edu\n' > ~/.paperscraper_env
chmod 600 ~/.paperscraper_env
```

## 7. Submit the scrape

```bash
sbatch examples/sol_gaudi/scrape_gaudi.sbatch
```

[`scrape_gaudi.sbatch`](../../examples/sol_gaudi/scrape_gaudi.sbatch) requests one Gaudi card, starts vLLM
from the shared environment in a background subshell, waits for `/v1/models` to
answer, then runs `ps_scrape` and `ps_store` from your environment against
`127.0.0.1`. The server is killed on exit.

Override anything at submit time:

```bash
sbatch --export=ALL,PS_RECIPE=polymer_db,PS_COUNT=1 examples/sol_gaudi/scrape_gaudi.sbatch
```

Watch it with `tail -f ps-gaudi-<jobid>.log`.

**Start with `PS_COUNT=1`.** A single paper exercises prompt construction, the
chat call, and JSON parsing end to end, and fails in about a minute instead of
four hours.

### Choosing a recipe

Both polymer recipes carry the biodegradation fields; they differ in how much
else they drag along, and that difference is a token budget question.

| | `polymer` (default) | `polymer_db` |
| --- | --- | --- |
| Fields | 36 | 86 |
| Degradation-related | 8 | 15 |
| Output tokens per record | ~450 | ~1100 |
| Records before truncation | ~22 | ~9 |
| Unit conversions in `ps_store` | 12 | 30 |

`polymer` covers the test standard, medium, extent, duration, mechanism,
degrading organisms, and certification — enough to reproduce something like
`tests/data/verification_data/bio.csv`, which is keyed on substance, guideline
(OECD 301B, 301D, …), and molecular weight.

`polymer_db` adds the full OECD 301/310 setup, mineralisation kinetics, and the
in vitro acid, base, hydrolytic, and enzymatic degradation curves. The cost is
that responses are capped at 10000 tokens, so a paper reporting a long sample
series truncates mid-JSON and loses the tail. Prefer it for papers with few
samples and deep characterisation.

Each recipe writes its own files — `materials_polymer.csv`,
`materials_polymer_db.csv` — because `ps_store` reuses an existing output file's
columns for matching, so two recipes sharing one CSV will not work.

Note that `ps_store`'s unit conversions are per unit-bearing column, not per row:
`polymer_db` costs 30 model round trips even for a single scraped record.

The script configures the model through `PAPERSCRAPER_MODEL_*` environment
variables rather than `ps_model_config`, because `ps_model_config` persists to
`~/.config/.pscraperrc.json` and concurrent jobs would race over it.

### Checking the server by hand

If a run misbehaves, get an interactive Gaudi session, start the server, and
separate server problems from client problems before involving PaperScraper:

```bash
curl -s "http://127.0.0.1:$PORT/v1/models"
```

The returned `data[0].id` must be exactly the Hugging Face repo id — that string
is what PaperScraper uses to load its tokenizer.

Then send the completion budget PaperScraper actually uses. This is the single
most useful check, because it is the one that catches the context-sizing mistake:

```bash
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$PS_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":10000}"
```

A 400 here means `--max-model-len` is too small, and every scrape request would
have failed the same way — silently, one paper at a time.

## 8. Scaling up

The default is deliberately conservative: one card, a 7B text model, 32k context.
Climb one rung at a time, confirming each works before the next.

1. **Text, 1 card** — the default. `Qwen/Qwen2.5-7B-Instruct`.
2. **Text, 8 cards** — a 72B-class model. Change `-G 1` to `-G 8`, uncomment
   `PT_HPU_ENABLE_LAZY_COLLECTIVES`, and add `--tensor-parallel-size 8`.
3. **Vision** — see below.
4. **Qwen3-VL** — the model in the CUDA notebook. Its Gaudi support is documented
   against Gaudi **3**, and Sol has Gaudi **2**. Check what the shared environment
   actually ships (`source activate gaudi-pytorch-vllm && pip show vllm`) before
   spending queue time on it.

### Context sizing — the one thing to get right

`PS_MAX_MODEL_LEN` and `PS_INPUT_TOKEN_LIMIT` are **not** the same number.

`paperscraper/extract.py` requests a 10000-token completion on every call, and
that value is hardcoded — there is no flag for it. vLLM rejects any request where
prompt plus completion exceeds `--max-model-len`. So the input budget has to
leave room for it:

```
PS_MAX_MODEL_LEN  >=  PS_INPUT_TOKEN_LIMIT + 11000
```

| `PS_MAX_MODEL_LEN` | `PS_INPUT_TOKEN_LIMIT` | Use |
| --- | --- | --- |
| 8192 | — | too small for any mode; every request 400s |
| 16384 | 5000 | `--mode abstract` only |
| **32768** | **20000** | **default; full text in a few chunks** |
| 65536 | 52000 | most papers in one chunk |

Setting the two equal is the most likely way to lose a four-hour allocation:
`scrape_gaudi.sbatch` refuses to launch if the gap is too small.

Note that the top row is ASU's own default — their Gaudi examples all serve at
`--max-model-len 8192`. Copy that number and nothing works, because the
hardcoded 10000-token completion does not fit in it on its own. This is the one
place these scripts deliberately depart from ASU's, and the memory knobs below
are how they pay for it.

### Gaudi launch flags

Beyond the context length, the scripts pass the three flags ASU's Gaudi examples
use. They are not vLLM defaults, and they are HPU-specific:

| Knob | Default | Why |
| --- | --- | --- |
| `PS_BLOCK_SIZE` | `128` | the KV block size the Gaudi backend is tuned for |
| `PS_GPU_MEMORY_UTILIZATION` | `0.80` | leaves device memory free for HPU graph capture |
| `PS_MAX_NUM_SEQS` | `16` | caps concurrent sequences, and so KV cache growth |

If warmup dies on an allocation failure, lower `PS_GPU_MEMORY_UTILIZATION`
before touching anything else:

```bash
sbatch --export=ALL,PS_GPU_MEMORY_UTILIZATION=0.70 examples/sol_gaudi/scrape_gaudi.sbatch
```

### Adding vision

PaperScraper keeps separate `text` and `vision` profiles, so the text model does
not have to change. Serve `Qwen/Qwen2.5-VL-7B-Instruct` (multimodal support for
it is enabled in the Gaudi plugin) and set the vision profile:

```bash
export PAPERSCRAPER_VISION_MODEL_PROVIDER=local
export PAPERSCRAPER_VISION_MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct
export PAPERSCRAPER_VISION_MODEL_BASE_URL="http://127.0.0.1:${PORT}/v1"
export PAPERSCRAPER_VISION_MODEL_CAPABILITIES=text,vision
```

Then run with `PS_MODE=text-images`. If you raise `ps_scrape --image-batch-size`
above its default of `1`, the server needs a matching
`--limit-mm-per-prompt image=N` or it will reject the requests.

## Troubleshooting

**The job sits at "Waiting for vLLM" for a long time.** Normal on first run. HPU
graph compilation happens during warmup, and weights may still be downloading.
Raise `PS_STARTUP_TIMEOUT`, or uncomment `VLLM_SKIP_WARMUP=true` while iterating —
but leave warmup on for real runs, since it buys steady-state throughput.

**vLLM exits during startup.** Almost always the model: unsupported architecture
on Gaudi 2, or it does not fit. Try a smaller model, or lower
`PS_MAX_MODEL_LEN`, `PS_GPU_MEMORY_UTILIZATION`, or `PS_MAX_NUM_SEQS` — see
Gaudi launch flags above.

**vLLM exits during startup with a PyTorch version mismatch.** The log shows
`Failed to load plugin hpu` and `AssertionError: Error: Compile-time major/minor
PyTorch version 2.7 differs from run-time 2.11.0+cu130`, then dies with
`RuntimeError: operator torchvision::nms does not exist`. One cause behind both:
a torch wheel under `~/.local/lib/python3.12/site-packages` is being imported
instead of the shared environment's Habana build, and the environment's
torchvision — compiled against the Habana build — then fails to register its
operators. Conda environments, unlike venvs, keep user site on `sys.path` *ahead*
of their own, so a single `pip install torch` run with no environment active
breaks every Gaudi job from then on. The job scripts set `PYTHONNOUSERSITE=1` and
check the resolved paths before loading the model, but clear the stray copy too —
anything else importing torch on this account hits the same problem:

```bash
ls ~/.local/lib/python3.12/site-packages | grep -iE 'torch|nvidia|triton'
python3.12 -m pip uninstall -y torch torchvision torchaudio
```

Run the uninstall with no environment active, so pip targets user site. If pip
reports the packages are not installed, that interpreter is not the one that owns
the directory — move the offending package directories aside by hand instead.

**Chunking looks wrong and nothing is logged.** PaperScraper falls back to a
`len/3` character estimate whenever the tokenizer fails to load, and it does so
silently. Confirm `PS_MODEL` is the exact Hugging Face repo id and that `HF_HOME`
is set in the job.

**The scrape produces no rows, but the job "succeeded".** This is the failure
mode to internalise. `ps_scrape` catches exceptions per paper, records them on
the row, and moves on, so a run where *every* request failed still exits 0 and
reports "0 material rows written". **Never judge a run by its exit code.** The
script prints a grouped dump of the recorded `last_error` values at the end —
read that. Common causes, in order:

- context-length 400s (see the sizing rule above),
- unparseable JSON from a model too small for the recipe — `polymer` has 36
  fields and `polymer_db` has 86, and a paper reporting a long sample series can
  hit the 10000-token completion ceiling and truncate mid-object; dropping from
  `polymer_db` to `polymer` is the first thing to try,
- a tokenizer that never loaded, so chunks were sized by character estimate.

**`ps_download` fails from a compute node.** Sol sits behind static NAT so
outbound access is expected to work, but if it does not, run step 6 on a login
node and submit only `papers.db` into the job. The corpus is self-contained.

**The job hangs after scraping.** `ps_store` prompts for confirmation without
`--assume-yes`. The provided script passes it.

## Before committing changes to these scripts

The scripts use `-p gaudi -q public -N 1 -G 1 -c 18`, matching ASU's
[vLLM page](https://docs.rc.asu.edu/vllm/) line for line, including `-G` rather
than `--gres`. Other ASU pages show `--partition=sol-gaudi --gres=gaudi:1` and
`--gres=gpu:hl225:8`; treat the vLLM page as authoritative for this workload, and
if a submit is rejected, run `sinfo -s` before editing anything.

`serve_gaudi.sbatch` is the one deliberate exception: it asks for `-t 0-08:00:00`
where ASU's examples use `-t 0-4`, because a server outliving several client jobs
is the whole point of it. A rejected submit means the QOS disagrees; lower it.
