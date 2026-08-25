#!/bin/bash
# Scrape biodegradable polymer data with a local vLLM server on four Gaudi 2 cards.
#
# The job starts vLLM from ASU Research Computing's shared gaudi-pytorch-vllm
# environment, waits for it to serve, then runs pmt scrape from our own
# paperminertoolkit environment against 127.0.0.1. Both live in the same job, so
# nothing is left running and no cross-node networking is involved.
#
# Submit from this directory, where fetch_corpus.sh left papers.db:
#
#   cd examples/sol_gaudi
#   sbatch scrape_gaudi.sbatch
#
# The log (ps-gaudi-<jobid>.log) and the CSVs land here too. Point PMT_DB at an
# absolute path if the corpus lives somewhere else.
#
# Every setting below can be overridden at submit time, for example:
#
#   sbatch --export=ALL,PMT_RECIPE=polymer_db,PMT_COUNT=5 scrape_gaudi.sbatch

#SBATCH --job-name=ps-scrape-gaudi
#SBATCH -p gaudi
#SBATCH -q public
#SBATCH -N 1
#SBATCH -G 4
#SBATCH -c 18
#SBATCH -t 0-04:00:00
#SBATCH -o ps-gaudi-%j.log

set -euo pipefail

# Run from wherever the job was submitted — normally examples/sol_gaudi — so the
# corpus and the CSVs are found and written beside the scripts rather than in the
# scheduler's spool directory.
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# --- Knobs ------------------------------------------------------------------
# PMT_MODEL must be the exact Hugging Face repo id. PaperMinerToolkit loads the
# matching tokenizer by that name for chunk sizing, so a mismatch silently
# degrades to a character estimate.
#
# Qwen3-30B-A3B-Instruct-2507 is the only Qwen3 model on Intel's validated list
# for Gaudi 2, and it is validated at tensor parallel 4 or 8 - hence -G 4 above.
# The -Instruct-2507 suffix is load-bearing: that variant is non-thinking by
# design. A hybrid thinking model (Qwen3-8B, any -Thinking- variant) spends the
# hardcoded 10000-token completion budget on reasoning before it writes any
# JSON, so records truncate or disappear entirely.
#
# 30.5B parameters at bf16 is ~61 GB of weights, ~15 GB per card across four,
# leaving the rest of each 96 GB card for KV cache and HPU graph capture.
PMT_MODEL="${PMT_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"

# Must equal the number of cards in the -G header above. ASU's own multi-card
# example is -G 8 -c 18 with --tensor-parallel-size 8, so the core count does
# not scale with the cards - only -G and this do. If the four-card queue is
# slow, drop to one card and a smaller model at submit time:
#
#   sbatch -G 1 --export=ALL,PMT_TENSOR_PARALLEL_SIZE=1,PMT_MODEL=Qwen/Qwen2.5-7B-Instruct \
#          scrape_gaudi.sbatch
PMT_TENSOR_PARALLEL_SIZE="${PMT_TENSOR_PARALLEL_SIZE:-4}"

# These two are NOT equal, and that is deliberate.
#
# paperminertoolkit/extract.py hardcodes max_output_tokens=10000 on every request,
# which OpenAICompatibleChatClient passes through as max_tokens. vLLM rejects a
# request when prompt_tokens + max_tokens exceeds --max-model-len. So the input
# budget must leave room for that fixed 10000-token completion:
#
#     PMT_MAX_MODEL_LEN >= PMT_INPUT_TOKEN_LIMIT + 11000
#
# Set them equal and every single request comes back HTTP 400.
#
# ASU's own Gaudi examples serve at --max-model-len 8192, which is unusable here:
# a single 10000-token completion does not fit inside it, so every request would
# 400 before the prompt is even counted. Hence 32768, paid for with the memory
# knobs below.
PMT_MAX_MODEL_LEN="${PMT_MAX_MODEL_LEN:-32768}"
PMT_INPUT_TOKEN_LIMIT="${PMT_INPUT_TOKEN_LIMIT:-20000}"
PMT_OUTPUT_RESERVE="${PMT_OUTPUT_RESERVE:-11000}"

# Gaudi-specific launch flags, taken from https://docs.rc.asu.edu/vllm/. None of
# them are vLLM defaults and all three matter on HPU: 128 is the block size the
# Gaudi backend is tuned for, and holding utilisation and concurrent sequences
# down leaves device memory free for HPU graph capture during warmup. Lower
# PMT_GPU_MEMORY_UTILIZATION first if warmup dies on an allocation failure.
PMT_BLOCK_SIZE="${PMT_BLOCK_SIZE:-128}"
PMT_GPU_MEMORY_UTILIZATION="${PMT_GPU_MEMORY_UTILIZATION:-0.80}"
PMT_MAX_NUM_SEQS="${PMT_MAX_NUM_SEQS:-16}"

PMT_DB="${PMT_DB:-papers.db}"
PMT_MODE="${PMT_MODE:-text}"
# Empty means every paper in the corpus, which is the intended setting: the
# scrape is resumable, so a run that hits the wall clock is continued simply by
# submitting again, and finished papers are skipped in seconds.
#
# Do not set PMT_COUNT to resume a partial run. --count slices the *ordered
# corpus*, not the unfinished part of it, so a second job with --count 50 would
# re-select the same first 50 papers, find them already succeeded, and do no new
# work. It is for testing a pipeline (PMT_COUNT=1), not for pacing a long run.
PMT_COUNT="${PMT_COUNT:-}"

# polymer: 36 fields, including the 8 biodegradation ones (test standard,
#   medium, extent, duration, mechanism, organisms, certification). ~450 output
#   tokens per record, so a sample series fits inside the 10000-token response
#   ceiling. It stays the default whatever the model: the ceiling belongs to the
#   pipeline, not to the model serving it.
# polymer_db: 86 fields, adding the full OECD 301/310 setup and the in vitro
#   acid/base/hydrolytic/enzymatic degradation curves. ~1100 tokens per record,
#   so a paper truncates after roughly 9 records. Use it when you need the whole
#   schema and the papers report few samples each.
PMT_RECIPE="${PMT_RECIPE:-polymer}"

# One materials file per recipe. pmt store reuses an existing output file's
# columns for matching, so two recipes must never share one CSV.
PMT_IN_CSV="${PMT_IN_CSV:-scraped_${PMT_RECIPE}.csv}"
PMT_OUT_CSV="${PMT_OUT_CSV:-materials_${PMT_RECIPE}.csv}"
PMT_ENV="${PMT_ENV:-paperminertoolkit}"    # our env, created per examples/sol_gaudi/README.md
PMT_VLLM_ENV="${PMT_VLLM_ENV:-gaudi-pytorch-vllm}"   # ASU's shared env - never install into it
PMT_RUN_STORE="${PMT_RUN_STORE:-1}"   # pmt store also calls the model, so do it while the server is up
PMT_STARTUP_TIMEOUT="${PMT_STARTUP_TIMEOUT:-3600}"

# --- Gaudi and cache environment -------------------------------------------
# Conda environments, unlike venvs, keep ~/.local/lib/pythonX.Y/site-packages on
# sys.path, ahead of the environment's own. One `pip install torch` made there
# with nothing activated shadows Habana's patched build for every later job, and
# the only symptom is an AssertionError deep inside vllm_gaudi about a
# compile-time/run-time PyTorch mismatch. Ignore user site for the whole job.
export PYTHONNOUSERSITE=1
export PT_HPU_LAZY_MODE=1
export HF_HOME="${HF_HOME:-/scratch/$USER/hf}"
mkdir -p "$HF_HOME"

# Uncomment while iterating to skip HPU graph warmup. Warmup is slow but buys
# steady-state throughput, so leave it on for real runs.
# export VLLM_SKIP_WARMUP=true

# Required for multi-card tensor parallelism: ASU's vLLM page states outright
# that collective communication across HPUs needs it. It is harmless on a single
# card, so it is set unconditionally rather than tied to the tensor parallel size.
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true

echo "=== Job ${SLURM_JOB_ID:-interactive} on $(hostname) ==="
echo "working directory: $PWD"
echo "model=$PMT_MODEL tp=$PMT_TENSOR_PARALLEL_SIZE max_model_len=$PMT_MAX_MODEL_LEN mode=$PMT_MODE recipe=$PMT_RECIPE"
echo "block_size=$PMT_BLOCK_SIZE gpu_mem_util=$PMT_GPU_MEMORY_UTILIZATION max_num_seqs=$PMT_MAX_NUM_SEQS"
hl-smi || echo "WARNING: hl-smi unavailable - are we really on a Gaudi node?"

if [[ ! -f "$PMT_DB" ]]; then
  echo "ERROR: $PMT_DB not found in $PWD." >&2
  echo "       Build the corpus first, from examples/sol_gaudi:" >&2
  echo "         ./fetch_corpus.sh          # or: sbatch fetch_corpus.sbatch" >&2
  echo "       Or set PMT_DB to the path of an existing corpus." >&2
  exit 1
fi

# Fail here rather than after a 30-minute model load.
if (( PMT_MAX_MODEL_LEN < PMT_INPUT_TOKEN_LIMIT + PMT_OUTPUT_RESERVE )); then
  echo "ERROR: PMT_MAX_MODEL_LEN ($PMT_MAX_MODEL_LEN) must be at least" >&2
  echo "       PMT_INPUT_TOKEN_LIMIT ($PMT_INPUT_TOKEN_LIMIT) + $PMT_OUTPUT_RESERVE." >&2
  echo "       PaperMinerToolkit always requests a 10000-token completion, and vLLM counts" >&2
  echo "       prompt + completion against --max-model-len. Every request would 400." >&2
  exit 1
fi

# Same reasoning for the card count. vLLM shards across exactly
# --tensor-parallel-size devices, so a mismatch with the -G allocation either
# dies during device discovery or hangs on the first collective - both minutes
# in, after the weights have started loading. SLURM_GPUS_ON_NODE is the
# allocation as the scheduler granted it; hl-smi is not consulted, because on a
# shared node it can report cards this job was not given.
if [[ -n "${SLURM_GPUS_ON_NODE:-}" && "$SLURM_GPUS_ON_NODE" =~ ^[0-9]+$ ]] \
   && (( PMT_TENSOR_PARALLEL_SIZE != SLURM_GPUS_ON_NODE )); then
  echo "ERROR: PMT_TENSOR_PARALLEL_SIZE ($PMT_TENSOR_PARALLEL_SIZE) does not match the" >&2
  echo "       $SLURM_GPUS_ON_NODE card(s) this job was allocated." >&2
  echo "       Change the -G line in this script, or set both at submit time:" >&2
  echo "         sbatch -G $PMT_TENSOR_PARALLEL_SIZE scrape_gaudi.sbatch" >&2
  exit 1
fi

# Module and conda activation scripts routinely reference unset variables, which
# would abort the job under `set -u`. Relax it around them only.
activate_env() {
  set +u
  source activate "$1"
  set -u
}

set +u
module load mamba/latest
set -u

PORT="$(shuf -i 8000-9000 -n 1)"
BASE_URL="http://127.0.0.1:${PORT}/v1"

# --- Preflight: torch must come from the vLLM environment -------------------
# PYTHONNOUSERSITE covers ~/.local, but PYTHONPATH or a stray .pth file shadows
# torch just as well. Check before paying for a model load: the mismatch only
# surfaces once vllm_gaudi imports habana_frameworks, minutes into a cold start,
# and the traceback that follows blames torchvision rather than the cause.
echo "=== Checking the vLLM environment ==="
(
  activate_env "$PMT_VLLM_ENV"
  python -u - <<'PY'
import importlib.metadata as metadata
import importlib.util
import pathlib
import sys

prefix = pathlib.Path(sys.prefix).resolve()
shadowed = []
for name in ("torch", "torchvision", "habana_frameworks"):
    # find_spec locates a package without importing it, so a mismatched pair is
    # reported here rather than raising on the way in.
    spec = importlib.util.find_spec(name)
    if spec is None:
        sys.exit(f"ERROR: {name} is not importable in {prefix}.")
    if spec.origin and spec.origin != "namespace":
        locations = [pathlib.Path(spec.origin).resolve().parent]
    else:
        # habana_frameworks is a namespace package: no top-level __init__.py, so
        # no origin, only the directories the package spans.
        locations = [pathlib.Path(p).resolve() for p in (spec.submodule_search_locations or [])]
    if not locations:
        sys.exit(f"ERROR: {name} is not importable in {prefix}.")
    try:
        version = metadata.version(name)
    except metadata.PackageNotFoundError:
        version = "?"
    shown = ", ".join(str(location) for location in locations)
    print(f"{name} {version} from {shown}")
    if any(prefix not in location.parents for location in locations):
        shadowed.append(name)

if shadowed:
    sys.exit(
        f"ERROR: shadowed by another install: {', '.join(shadowed)}\n"
        f"       These resolve outside {prefix}, so the Habana build is not the\n"
        "       one that loads and vLLM will fail during startup. Fix it with\n"
        "         python -m pip uninstall -y torch torchvision torchaudio\n"
        "       run with no environment active, or move the offending directory\n"
        "       aside. Never install into the shared vLLM environment itself."
    )
PY
)

# --- Start vLLM -------------------------------------------------------------
# The subshell is load-bearing: activating our env later in the parent shell
# would otherwise clobber the interpreter the server is running under.
(
  activate_env "$PMT_VLLM_ENV"
  exec python -m vllm.entrypoints.openai.api_server \
    --model "$PMT_MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --tensor-parallel-size "$PMT_TENSOR_PARALLEL_SIZE" \
    --max-model-len "$PMT_MAX_MODEL_LEN" \
    --block-size "$PMT_BLOCK_SIZE" \
    --gpu-memory-utilization "$PMT_GPU_MEMORY_UTILIZATION" \
    --max-num-seqs "$PMT_MAX_NUM_SEQS"
) &
VLLM_PID=$!

cleanup() {
  echo "=== Stopping vLLM (pid $VLLM_PID) ==="
  kill "$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

# --- Wait for readiness -----------------------------------------------------
echo "=== Waiting for vLLM on port $PORT (timeout ${PMT_STARTUP_TIMEOUT}s) ==="
deadline=$(( SECONDS + PMT_STARTUP_TIMEOUT ))
until curl -sf "${BASE_URL}/models" >/dev/null 2>&1; do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "ERROR: vLLM exited during startup. Check the log above for the model load failure." >&2
    exit 1
  fi
  if (( SECONDS > deadline )); then
    echo "ERROR: vLLM did not become ready within ${PMT_STARTUP_TIMEOUT}s." >&2
    echo "First load compiles HPU graphs and can be slow; raise PMT_STARTUP_TIMEOUT or set VLLM_SKIP_WARMUP=true." >&2
    exit 1
  fi
  sleep 10
done
echo "=== vLLM ready ==="

# --- Run PaperMinerToolkit -------------------------------------------------------
activate_env "$PMT_ENV"

# Configure the text profile through environment overrides rather than
# pmt config model. pmt config model persists to ~/.config/.paperminertoolkitrc.json,
# which concurrent jobs would race over; these variables are job-local.
export PAPERMINERTOOLKIT_MODEL_PROVIDER=local
export PAPERMINERTOOLKIT_MODEL_NAME="$PMT_MODEL"
export PAPERMINERTOOLKIT_MODEL_BASE_URL="$BASE_URL"
export PAPERMINERTOOLKIT_MODEL_CAPABILITIES=text
export PAPERMINERTOOLKIT_MODEL_INPUT_TOKEN_LIMIT="$PMT_INPUT_TOKEN_LIMIT"

pmt config status

scrape_args=(--mode "$PMT_MODE" --output "$PMT_IN_CSV")
[[ -n "$PMT_COUNT" ]] && scrape_args+=(--count "$PMT_COUNT")

echo "=== Scraping ==="
pmt scrape "$PMT_DB" "$PMT_RECIPE" "${scrape_args[@]}"

if [[ "$PMT_RUN_STORE" == "1" ]]; then
  # pmt store runs LLM-driven unit conversion, so it needs the server too.
  # --assume-yes is required: without it pmt store waits on an interactive
  # confirmation and the job would sit idle until it hits the walltime.
  echo "=== Storing ==="
  pmt store "$PMT_DB" "$PMT_IN_CSV" "$PMT_OUT_CSV" "$PMT_RECIPE" --assume-yes
fi

pmt status "$PMT_DB"

# pmt scrape catches per-paper exceptions, records them on the row, and carries
# on, so a completely broken run still exits 0 and reports "0 material rows".
# Surface the recorded errors explicitly or they go unnoticed.
echo "=== Recorded per-paper errors ==="
python - "$PMT_DB" <<'PY'
import sqlite3, sys
con = sqlite3.connect(sys.argv[1])
rows = con.execute(
    "SELECT substr(last_error, 1, 200) AS err, COUNT(*) AS n"
    " FROM papers WHERE last_error IS NOT NULL AND last_error != ''"
    " GROUP BY err ORDER BY n DESC LIMIT 10"
).fetchall()
if not rows:
    print("none")
for err, n in rows:
    print(f"[{n}x] {err}")
PY

# --- What is left ----------------------------------------------------------
# A corpus large enough to need this script is larger than one four-hour
# allocation, so the normal end state is "partly done". Say so explicitly, and
# print the command that continues it.
echo "=== Remaining work ==="
python - "$PMT_DB" "$PMT_MODE" <<'PY'
import sqlite3
import sys

db_path, mode = sys.argv[1], sys.argv[2]
stages = {
    'abstract': ['abstract_scrape_status'],
    'text': ['text_scrape_status'],
    'images': ['image_scrape_status'],
    'text-images': ['text_scrape_status', 'image_scrape_status'],
}[mode]

con = sqlite3.connect(db_path)
total = con.execute('SELECT COUNT(*) FROM papers').fetchone()[0]
remaining = 0
for column in stages:
    # column comes from the table above, never from user input.
    done = con.execute(
        f"SELECT COUNT(*) FROM papers WHERE COALESCE({column}, '') = 'succeeded'"
    ).fetchone()[0]
    print(f'{column}: {done} of {total} succeeded, {total - done} to go')
    remaining = max(remaining, total - done)

if remaining:
    print()
    print(f'{remaining} papers still to scrape. The scrape is resumable and skips')
    print('what already succeeded, so continue it by submitting the same job again:')
    print(f'  sbatch --export=ALL,PMT_DB={db_path} scrape_gaudi.sbatch')
else:
    print()
    print('Corpus fully scraped.')
PY

echo "=== Done ==="
