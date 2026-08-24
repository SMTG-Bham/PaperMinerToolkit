# Running on ASU Sol's Intel Gaudi nodes

This guide gets PaperMiner running against a local vLLM server on Sol's Gaudi 2
accelerators. It replaces the general
[`qwen_vllm_workflow.ipynb`](../../docs/examples/qwen_vllm_workflow.ipynb),
which assumes a CUDA workstation.

The worked example is biodegradable polymers: the scripts default to the
`polymer` recipe and an OECD 301 search query. Everything here applies to any
recipe — override `PS_RECIPE` and the query if you are after something else.

The important thing to understand first: **PaperMiner does no accelerator work
of its own.** It is an HTTP client that speaks the OpenAI chat API. The Gaudi
cards are used only by a separate vLLM server process. So this is not a code
port — it is two environments and an sbatch script.

Those two environments cannot be merged. The vLLM Gaudi stack is built on Python
3.10; PaperMiner requires 3.11 or newer. They talk over `localhost` instead.

| Environment | Owner | Python | Contents |
| --- | --- | --- | --- |
| `gaudi-pytorch-vllm` | ASU Research Computing (shared) | 3.10 | vLLM + Habana PyTorch |
| `paperminer` | you | 3.14 | PaperMiner and its dependencies |

> **Never `pip install` into `gaudi-pytorch-vllm`.** It is shared by everyone on
> Sol, and pip would happily replace Habana's patched PyTorch with a stock wheel,
> silently breaking HPU support for every other user. Only ever `source activate`
> it.

The runnable scripts live in this directory, alongside this guide:

| Script | Runs on | Purpose |
| --- | --- | --- |
| [`install.sbatch`](install.sbatch) | `htc` | **Run this first.** Builds the environment and verifies it |
| [`fetch_corpus.sh`](fetch_corpus.sh) | login node or interactive | Search and download; never touches the model |
| [`fetch_corpus.sbatch`](fetch_corpus.sbatch) | `htc` | The same thing as a batch job, for corpora too big to babysit |
| [`scrape_gaudi.sbatch`](scrape_gaudi.sbatch) | `gaudi`, 4 cards | **Start here.** vLLM + scrape + store in one job |
| [`serve_gaudi.sbatch`](serve_gaudi.sbatch) | `gaudi`, 4 cards | Long-lived server for reusing a warm model |

Submit them in place, or copy them somewhere else and edit — every setting is an
environment variable with a default.

Everything in this example is submitted from **this directory**, and everything
it produces is written back here: the job logs, `papers.db`, and the CSVs. Slurm
runs a job in the directory it was submitted from, so `cd examples/sol_gaudi`
once and the paths take care of themselves.

## Quick start

From this directory, three submissions end to end:

```bash
cd examples/sol_gaudi
sbatch install.sbatch
sbatch fetch_corpus.sbatch
sbatch scrape_gaudi.sbatch
```

`install.sbatch` is the one job that has to *run* from the repository root, for
`build_tools/environment.yml` and the editable install; it walks up from the
submit directory to find the checkout, so it is still submitted from here like
the rest. The other two work in the directory they were submitted from, which is
why `papers.db` and the CSVs end up beside the scripts.

Wait for each to finish before the next — the corpus has to exist before the
scrape, and the environment before either. The logs land here too, so watch
progress with `tail -f ps-install-<jobid>.log`.

Sections 1 to 4 explain what `install.sbatch` does and how to do it by hand;
skip to section 5 if the install job succeeded. Those by-hand commands run from
the repository root, since they install the checkout — only the submissions run
from here.

## 1. Get an interactive session

Only needed if you are installing by hand. ASU asks that environments not be
built on a login node, which is why `install.sbatch` is a batch job.

```bash
interactive -c 4 -t 0-2
```

That is shorthand for `salloc -c 1 -p htc -q public -t 0-4` with your overrides.

## 2. Create the PaperMiner environment

```bash
module load mamba/latest
mamba env create -f build_tools/environment.yml
source activate paperminer
```

[`build_tools/environment.yml`](../../build_tools/environment.yml) supplies only the interpreter;
every dependency comes from `pyproject.toml`. Use `source activate`, never
`mamba activate` — the latter writes cruft into your shell configuration.

`install.sbatch` does this for you, and **rebuilds from scratch by default**: if
`$PS_ENV` already exists it is removed and recreated, so every run gives the same
environment regardless of what the last one left behind. Pass `PS_FORCE=0` to
reuse an existing environment instead:

```bash
sbatch --export=ALL,PS_FORCE=0 install.sbatch
```

It only ever touches `$PS_ENV` — no other environment on the account — and
refuses outright if that name resolves to a prefix you do not own, which is what
keeps a typo away from ASU's shared environments.

## 3. Install the package, without dragging in CUDA

ASU's [vLLM page](https://docs.rc.asu.edu/vllm/) tells you to use the `mamba`
module or an apptainer image, and — verbatim — "Do not use `uv`, `conda`, or
`pip`, or docker." That rule is about building your own inference stack, and the
PyTorch-mismatch entry under Troubleshooting is exactly what it exists to
prevent. What follows stays inside it: the environment comes from the `mamba`
module, torch and vLLM are only ever the shared `gaudi-pytorch-vllm` copies, and
`pip` installs nothing but PaperMiner and its pure-Python dependencies into an
environment of our own — never the shared one, never `~/.local`. If your group
reads the rule more strictly than that, build a `.sif` and run the scraper from
it instead; nothing in PaperMiner needs the host environment.

`pyproject.toml` depends on `headroom-ai[image,ml]`, whose `ml` extra requires
`torch`. Left alone, pip resolves that to the CUDA build and pulls several GB of
`nvidia-*` wheels and `triton` that are useless on a Gaudi node. PaperMiner only
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
sbatch --export=ALL,PS_PREFETCH_WEIGHTS=1 install.sbatch
```

## 4. Point caches at scratch

`/home` is capped at 100 GiB and model weights run 15–140 GB, so keep the
Hugging Face cache on scratch. Add to your `~/.bashrc`:

```bash
export HF_HOME="/scratch/$USER/hf"
export PIP_CACHE_DIR="/scratch/$USER/pip"
```

This matters on both sides. vLLM downloads the weights, and PaperMiner loads
the *tokenizer* for the same model name to size its text chunks — sharing one
`HF_HOME` means the second lookup is a cache hit rather than a re-download.

## 5. Set credentials

Searching and downloading need publisher credentials; the local model does not
need an OpenAI or Anthropic key.

```bash
export ELSEVIER_API_KEY="..."
export CORE_API_KEY="..."
export UNPAYWALL_EMAIL="you@asu.edu"
export OPENALEX_API_KEY="..."
```

The interactive `ps_elsevier_key`, `ps_core_key`, `ps_unpaywall_email`, and
`ps_openalex_key` commands store these in `~/.config/.pscraperrc.json` if you
prefer.

OpenAlex is the one source that still answers without a key, but it meters
requests against a daily credit budget that a keyless caller exhausts after
roughly 100 search pages. A free key from <https://openalex.org/settings/api>
raises that budget tenfold, which a full corpus build needs.

## 6. Build the corpus off the Gaudi queue

Search and download are network-bound and never call the model, so they must not
burn a Gaudi allocation.

```bash
cd examples/sol_gaudi
./fetch_corpus.sh
```

That uses the default query, `biodegradable polymer OECD 301 biodegradation`.
Pass your own as the first argument:

```bash
./fetch_corpus.sh "polymer biodegradation OECD 310" papers.db 100
```

Naming the test standard — OECD 301, OECD 310, ASTM D6400, ISO 14855 — tends to
surface papers with extractable results rather than reviews, which matters
because the biodegradation fields are only filled when a paper states them.

The third argument is the result cap, and it now defaults to **everything the
sources will return**. Two things to know about it:

- It is **per source, not in total.** `--source all` asks Scopus, CORE and
  OpenAlex for that many each, then merges on DOI, so the corpus lands between
  the largest single source and the sum of the three.
- There is no "unlimited" flag in `ps_search`. Each backend loops until it has
  the requested number or the provider runs out, so the default is simply a
  number larger than any real result set. Scopus stops at its own total, CORE
  and OpenAlex stop when a short page comes back.

Bound it with a number while you are testing a new query — a broad term can
return tens of thousands of papers, and `ps_download` then fetches all of them.

For a corpus large enough that you do not want to sit in an interactive session
waiting on rate limits, submit it instead:

```bash
sbatch fetch_corpus.sbatch
sbatch --export=ALL,PS_COUNT=500 fetch_corpus.sbatch
```

That runs on `htc` with no accelerator requested, since neither stage calls the
model. `htc` caps wall time at four hours but runs uninterrupted. Submit it from
this directory so `papers.db` lands beside the scripts rather than in the
scheduler's spool directory — a relative `PS_DB` is resolved against the
directory you submitted from, and `fetch_corpus.sh` anchors one to its own
directory, so both routes agree on where the corpus lives.

**The four-hour cap and an unbounded `PS_COUNT` interact badly on a large
corpus**, and the two halves of the job behave differently if it is killed:

- `ps_search` upserts on DOI, so re-running it costs little and adds nothing
  twice.
- `ps_download` walks every row in the corpus and has **no skip for assets it
  already holds** — only abstracts short-circuit. A job that dies at the wall
  clock re-downloads everything on the next submit.

So either bound `PS_COUNT` so the fetch finishes inside four hours, or switch the
header to `-p general -t 1-00:00:00`, which the `public` QOS allows up to a week.

A batch job does not reliably inherit your shell environment, so put the
credentials in a file only you can read and the script will source it:

```bash
printf 'export ELSEVIER_API_KEY=...\nexport CORE_API_KEY=...\nexport UNPAYWALL_EMAIL=you@asu.edu\nexport OPENALEX_API_KEY=...\n' > ~/.paperminer_env
chmod 600 ~/.paperminer_env
```

## 7. Submit the scrape

```bash
sbatch scrape_gaudi.sbatch
```

From this directory again, so the job finds `papers.db` where the corpus step
left it and writes the CSVs and its log alongside.

[`scrape_gaudi.sbatch`](scrape_gaudi.sbatch) requests four Gaudi cards, starts
vLLM from the shared environment in a background subshell, waits for
`/v1/models` to answer, then runs `ps_scrape` and `ps_store` from your
environment against `127.0.0.1`. The server is killed on exit.

Four cards because of the model. The default is
[`Qwen/Qwen3-30B-A3B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507),
the only Qwen3 model on Intel's
[validated list](https://docs.vllm.ai/projects/gaudi/en/latest/getting_started/validated_models.html)
for Gaudi **2** — every other Qwen3 entry is validated on Gaudi 3 — and it is
validated there at tensor parallel 4 or 8. `-G 4` and `--tensor-parallel-size 4`
have to agree; the scripts refuse to start if they do not.

If the four-card queue is slow, one card and the older model still work:

```bash
sbatch -G 1 --export=ALL,PS_TENSOR_PARALLEL_SIZE=1,PS_MODEL=Qwen/Qwen2.5-7B-Instruct scrape_gaudi.sbatch
```

Override anything at submit time:

```bash
sbatch --export=ALL,PS_RECIPE=polymer_db,PS_COUNT=1 scrape_gaudi.sbatch
```

Watch it with `tail -f ps-gaudi-<jobid>.log`.

**Start with `PS_COUNT=1`.** A single paper exercises prompt construction, the
chat call, and JSON parsing end to end, and fails in about a minute instead of
four hours.

### Scraping the whole corpus

`PS_COUNT` is empty by default, which means every paper in `papers.db`. A corpus
worth building is normally bigger than one four-hour Gaudi allocation, so expect
the job to be killed part-way through. That is fine, and it is the intended way
to run this:

- Every paper's outcome is committed to `papers.db` as it happens, so a job that
  hits the wall clock loses only the paper in flight.
- `ps_scrape` skips any stage already marked `succeeded`, so re-submitting the
  same job continues where the last one stopped. Finished papers cost
  milliseconds each.
- The job now ends with a **Remaining work** section: how many papers are done,
  how many are left, and the command to continue. Resubmit until it says
  `Corpus fully scraped.`

```bash
sbatch scrape_gaudi.sbatch   # repeat until nothing remains
```

**Do not use `PS_COUNT` to pace a long run.** `--count` slices the *ordered
corpus*, not the unfinished part of it, so a second job with `PS_COUNT=50` would
re-select the same first 50 papers, find them already succeeded, and do no new
work. Use it for testing (`PS_COUNT=1`), never for resuming.

The one thing to watch is the CSV: `ps_scrape --output` appends across runs, and
`ps_store` matches against the existing columns, so keep one file per recipe and
let successive jobs grow it.

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

The script configures the model through `PAPERMINER_MODEL_*` environment
variables rather than `ps_model_config`, because `ps_model_config` persists to
`~/.config/.pscraperrc.json` and concurrent jobs would race over it.

### Checking the server by hand

If a run misbehaves, get an interactive Gaudi session, start the server, and
separate server problems from client problems before involving PaperMiner:

```bash
curl -s "http://127.0.0.1:$PORT/v1/models"
```

The returned `data[0].id` must be exactly the Hugging Face repo id — that string
is what PaperMiner uses to load its tokenizer.

Then send the completion budget PaperMiner actually uses. This is the single
most useful check, because it is the one that catches the context-sizing mistake:

```bash
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$PS_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":10000}"
```

A 400 here means `--max-model-len` is too small, and every scrape request would
have failed the same way — silently, one paper at a time.

## 8. Scaling up

The default is four cards, a 30B-A3B text model, 32k context. Move one rung at a
time, confirming each works before the next.

1. **Text, 1 card** — `Qwen/Qwen2.5-7B-Instruct` at `PS_TENSOR_PARALLEL_SIZE=1`.
   The cheapest thing to get queued, and the right rung for a first end-to-end
   test with `PS_COUNT=1`.
2. **Text, 4 cards** — the default, `Qwen/Qwen3-30B-A3B-Instruct-2507`.
3. **Text, 8 cards** — the same model at `-G 8` and `PS_TENSOR_PARALLEL_SIZE=8`,
   which Intel also validated. More KV headroom rather than a bigger model, so
   spend it on `PS_MAX_MODEL_LEN` and `PS_MAX_NUM_SEQS`, not on the weights.
4. **Vision** — see below.

### Choosing a model

`PS_MODEL` has to satisfy three things at once, and the default is the model that
does.

**It has to be validated on Gaudi 2.** Sol's cards are Gaudi 2, and Intel's
[validated models list](https://docs.vllm.ai/projects/gaudi/en/latest/getting_started/validated_models.html)
is the thing to check before spending queue time. Of the Qwen3 family, exactly one
entry covers Gaudi 2 — `Qwen/Qwen3-30B-A3B-Instruct-2507`, at tensor parallel 4 or
8, BF16 or FP8. Everything else Qwen3, including all the Qwen3-VL entries, is
validated against Gaudi **3** only. That is also why the CUDA notebook's
`Qwen/Qwen3-VL-30B-A3B-Instruct` is not the default here.

**It must not be a thinking model.** `paperminer/extract.py` asks for a
10000-token completion and nothing more. A hybrid-thinking model — `Qwen3-8B`,
anything `-Thinking-` — spends that budget reasoning before it writes any JSON,
so records truncate or the response comes back with no JSON at all. The
`-Instruct-2507` variants are non-thinking by design, which is what makes this one
safe to point a rigid extraction pipeline at. Disabling thinking server-side needs
`--default-chat-template-kwargs '{"enable_thinking": false}'`, which only exists in
newer vLLM; do not assume the shared environment has it.

**It has to fit.** 30.5B parameters at BF16 is roughly 61 GB of weights, about
15 GB per card across four 96 GB Gaudi 2 cards, leaving the rest for KV cache and
HPU graph capture. Only 3.3B parameters are active per token, so throughput is
closer to a small dense model than the parameter count suggests.

The model card recommends `temperature=0.7, top_p=0.8` for general use, while
PaperMiner defaults to `temperature=0, top_p=1` so extraction is reproducible.
Keep the deterministic defaults; if you see a run degenerate into repetition that
eats the completion budget, `PAPERMINER_MODEL_TEMPERATURE` and
`PAPERMINER_MODEL_TOP_P` override them per job.

### Context sizing — the one thing to get right

`PS_MAX_MODEL_LEN` and `PS_INPUT_TOKEN_LIMIT` are **not** the same number.

`paperminer/extract.py` requests a 10000-token completion on every call, and
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

The default model supports 262144 tokens natively, so the ceiling here is device
memory rather than the model. At tensor parallel 4 the weights take about 15 GB
of each card, which leaves enough KV cache for 65536 to be worth trying — raise
`PS_MAX_MODEL_LEN` and `PS_INPUT_TOKEN_LIMIT` together, keeping the 11000-token
gap.

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
sbatch --export=ALL,PS_GPU_MEMORY_UTILIZATION=0.70 scrape_gaudi.sbatch
```

### Adding vision

PaperMiner keeps separate `text` and `vision` profiles, so the text model does
not have to change — but it does mean a second server, and `scrape_gaudi.sbatch`
has no spare cards to put one on. Run the vision model as its own job with
[`serve_gaudi.sbatch`](serve_gaudi.sbatch), on one card:

```bash
sbatch -G 1 --export=ALL,PS_TENSOR_PARALLEL_SIZE=1,PS_MODEL=Qwen/Qwen2.5-VL-7B-Instruct,ENDPOINT_FILE=$PWD/vllm_vision_endpoint.txt \
       serve_gaudi.sbatch
cat vllm_vision_endpoint.txt     # -> <node>:<port>
```

`ENDPOINT_FILE` keeps it clear of a text server started the same way: both
default to `vllm_endpoint.txt`, and the second job to start would overwrite the
first.

`Qwen/Qwen2.5-VL-7B-Instruct` is the vision model to reach for: it runs on one
card and its multimodal support is enabled in the Gaudi plugin. Intel validated it
on Gaudi 3, as it did every vision entry on that list, so on Sol's Gaudi 2 it is
supported rather than proven — try it on a couple of papers before committing a
run to it. Point the vision profile at the endpoint the job wrote:

```bash
export PAPERMINER_VISION_MODEL_PROVIDER=local
export PAPERMINER_VISION_MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct
export PAPERMINER_VISION_MODEL_BASE_URL="http://$(cat vllm_vision_endpoint.txt)/v1"
export PAPERMINER_VISION_MODEL_CAPABILITIES=text,vision
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

**The job dies immediately with a card-count error.** `PS_TENSOR_PARALLEL_SIZE`
and the `-G` header have to agree, and both scripts check that against
`SLURM_GPUS_ON_NODE` before loading anything. Change the header, or set both at
submit time: `sbatch -G 1 --export=ALL,PS_TENSOR_PARALLEL_SIZE=1 …`. Left
unchecked this failure costs minutes of model load and then hangs on the first
collective rather than exiting.

**Responses arrive full of reasoning and short on JSON.** The model is a thinking
one. `extract.py` asks for 10000 completion tokens and nothing more, so reasoning
comes out of the same budget the records need. Use an `-Instruct-` variant — see
Choosing a model above.

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

**Chunking looks wrong and nothing is logged.** PaperMiner falls back to a
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

The scripts use `-p gaudi -q public -N 1 -G 4 -c 18`, following ASU's
[vLLM page](https://docs.rc.asu.edu/vllm/), including `-G` rather than `--gres`.
The card count is the one number that differs from their single-card example, and
it tracks their multi-card one: ASU's 72B job is `-G 8 -c 18` with
`--tensor-parallel-size 8`, so cores do not scale with cards and `-c 18` stands.
`PT_HPU_ENABLE_LAZY_COLLECTIVES=true` is set for the same reason — that page
requires it for collectives across HPUs. Other ASU pages show
`--partition=sol-gaudi --gres=gaudi:1` and `--gres=gpu:hl225:8`; treat the vLLM
page as authoritative for this workload, and if a submit is rejected, run
`sinfo -s` before editing anything.

If you change `PS_MODEL`, check it against Intel's
[validated models list](https://docs.vllm.ai/projects/gaudi/en/latest/getting_started/validated_models.html)
for a Gaudi **2** entry and confirm the tensor parallel size it was validated at,
then update the `-G` header, `PS_TENSOR_PARALLEL_SIZE`, and the tokenizer default
in [`install.sbatch`](install.sbatch) together.

`serve_gaudi.sbatch` is the one deliberate exception: it asks for `-t 0-08:00:00`
where ASU's examples use `-t 0-4`, because a server outliving several client jobs
is the whole point of it. A rejected submit means the QOS disagrees; lower it.
