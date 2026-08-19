#!/bin/bash
# Scrape biodegradable polymer data with a local vLLM server on one Gaudi 2 card.
#
# The job starts vLLM from ASU Research Computing's shared gaudi-pytorch-vllm
# environment, waits for it to serve, then runs ps_scrape from our own
# paperscraper environment against 127.0.0.1. Both live in the same job, so
# nothing is left running and no cross-node networking is involved.
#
# Submit from this directory, where fetch_corpus.sh left papers.db:
#
#   cd examples/sol_gaudi
#   sbatch scrape_gaudi.sbatch
#
# The log (ps-gaudi-<jobid>.log) and the CSVs land here too. Point PS_DB at an
# absolute path if the corpus lives somewhere else.
#
# Every setting below can be overridden at submit time, for example:
#
#   sbatch --export=ALL,PS_RECIPE=polymer_db,PS_COUNT=5 scrape_gaudi.sbatch

#SBATCH --job-name=ps-scrape-gaudi
#SBATCH -p gaudi
#SBATCH -q public
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 18
#SBATCH -t 0-04:00:00
#SBATCH -o ps-gaudi-%j.log

set -euo pipefail

# Run from wherever the job was submitted — normally examples/sol_gaudi — so the
# corpus and the CSVs are found and written beside the scripts rather than in the
# scheduler's spool directory.
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# --- Knobs ------------------------------------------------------------------
# PS_MODEL must be the exact Hugging Face repo id. PaperScraper loads the
# matching tokenizer by that name for chunk sizing, so a mismatch silently
# degrades to a character estimate.
PS_MODEL="${PS_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

# These two are NOT equal, and that is deliberate.
#
# paperscraper/extract.py hardcodes max_output_tokens=10000 on every request,
# which OpenAICompatibleChatClient passes through as max_tokens. vLLM rejects a
# request when prompt_tokens + max_tokens exceeds --max-model-len. So the input
# budget must leave room for that fixed 10000-token completion:
#
#     PS_MAX_MODEL_LEN >= PS_INPUT_TOKEN_LIMIT + 11000
#
# Set them equal and every single request comes back HTTP 400.
#
# ASU's own Gaudi examples serve at --max-model-len 8192, which is unusable here:
# a single 10000-token completion does not fit inside it, so every request would
# 400 before the prompt is even counted. Hence 32768, paid for with the memory
# knobs below.
PS_MAX_MODEL_LEN="${PS_MAX_MODEL_LEN:-32768}"
PS_INPUT_TOKEN_LIMIT="${PS_INPUT_TOKEN_LIMIT:-20000}"
PS_OUTPUT_RESERVE="${PS_OUTPUT_RESERVE:-11000}"

# Gaudi-specific launch flags, taken from https://docs.rc.asu.edu/vllm/. None of
# them are vLLM defaults and all three matter on HPU: 128 is the block size the
# Gaudi backend is tuned for, and holding utilisation and concurrent sequences
# down leaves device memory free for HPU graph capture during warmup. Lower
# PS_GPU_MEMORY_UTILIZATION first if warmup dies on an allocation failure.
PS_BLOCK_SIZE="${PS_BLOCK_SIZE:-128}"
PS_GPU_MEMORY_UTILIZATION="${PS_GPU_MEMORY_UTILIZATION:-0.80}"
PS_MAX_NUM_SEQS="${PS_MAX_NUM_SEQS:-16}"

PS_DB="${PS_DB:-papers.db}"
PS_MODE="${PS_MODE:-text}"
# Empty means every paper in the corpus, which is the intended setting: the
# scrape is resumable, so a run that hits the wall clock is continued simply by
# submitting again, and finished papers are skipped in seconds.
#
# Do not set PS_COUNT to resume a partial run. --count slices the *ordered
# corpus*, not the unfinished part of it, so a second job with --count 50 would
# re-select the same first 50 papers, find them already succeeded, and do no new
# work. It is for testing a pipeline (PS_COUNT=1), not for pacing a long run.
PS_COUNT="${PS_COUNT:-}"

# polymer: 36 fields, including the 8 biodegradation ones (test standard,
#   medium, extent, duration, mechanism, organisms, certification). ~450 output
#   tokens per record, so a sample series fits inside the 10000-token response
#   ceiling. This is the right default on a 7B model.
# polymer_db: 86 fields, adding the full OECD 301/310 setup and the in vitro
#   acid/base/hydrolytic/enzymatic degradation curves. ~1100 tokens per record,
#   so a paper truncates after roughly 9 records. Use it when you need the whole
#   schema and the papers report few samples each.
PS_RECIPE="${PS_RECIPE:-polymer}"

# One materials file per recipe. ps_store reuses an existing output file's
# columns for matching, so two recipes must never share one CSV.
PS_IN_CSV="${PS_IN_CSV:-scraped_${PS_RECIPE}.csv}"
PS_OUT_CSV="${PS_OUT_CSV:-materials_${PS_RECIPE}.csv}"
PS_ENV="${PS_ENV:-paperscraper}"    # our env, created per examples/sol_gaudi/README.md
PS_VLLM_ENV="${PS_VLLM_ENV:-gaudi-pytorch-vllm}"   # ASU's shared env - never install into it
PS_RUN_STORE="${PS_RUN_STORE:-1}"   # ps_store also calls the model, so do it while the server is up
PS_STARTUP_TIMEOUT="${PS_STARTUP_TIMEOUT:-3600}"

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

# Multi-card only. Uncomment together with raising -G above 1, and add a
# matching --tensor-parallel-size to the launch below.
# export PT_HPU_ENABLE_LAZY_COLLECTIVES=true

echo "=== Job ${SLURM_JOB_ID:-interactive} on $(hostname) ==="
echo "working directory: $PWD"
echo "model=$PS_MODEL max_model_len=$PS_MAX_MODEL_LEN mode=$PS_MODE recipe=$PS_RECIPE"
echo "block_size=$PS_BLOCK_SIZE gpu_mem_util=$PS_GPU_MEMORY_UTILIZATION max_num_seqs=$PS_MAX_NUM_SEQS"
hl-smi || echo "WARNING: hl-smi unavailable - are we really on a Gaudi node?"

if [[ ! -f "$PS_DB" ]]; then
  echo "ERROR: $PS_DB not found in $PWD." >&2
  echo "       Build the corpus first, from examples/sol_gaudi:" >&2
  echo "         ./fetch_corpus.sh          # or: sbatch fetch_corpus.sbatch" >&2
  echo "       Or set PS_DB to the path of an existing corpus." >&2
  exit 1
fi

# Fail here rather than after a 30-minute model load.
if (( PS_MAX_MODEL_LEN < PS_INPUT_TOKEN_LIMIT + PS_OUTPUT_RESERVE )); then
  echo "ERROR: PS_MAX_MODEL_LEN ($PS_MAX_MODEL_LEN) must be at least" >&2
  echo "       PS_INPUT_TOKEN_LIMIT ($PS_INPUT_TOKEN_LIMIT) + $PS_OUTPUT_RESERVE." >&2
  echo "       PaperScraper always requests a 10000-token completion, and vLLM counts" >&2
  echo "       prompt + completion against --max-model-len. Every request would 400." >&2
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
  activate_env "$PS_VLLM_ENV"
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
  activate_env "$PS_VLLM_ENV"
  exec python -m vllm.entrypoints.openai.api_server \
    --model "$PS_MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --max-model-len "$PS_MAX_MODEL_LEN" \
    --block-size "$PS_BLOCK_SIZE" \
    --gpu-memory-utilization "$PS_GPU_MEMORY_UTILIZATION" \
    --max-num-seqs "$PS_MAX_NUM_SEQS"
) &
VLLM_PID=$!

cleanup() {
  echo "=== Stopping vLLM (pid $VLLM_PID) ==="
  kill "$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

# --- Wait for readiness -----------------------------------------------------
echo "=== Waiting for vLLM on port $PORT (timeout ${PS_STARTUP_TIMEOUT}s) ==="
deadline=$(( SECONDS + PS_STARTUP_TIMEOUT ))
until curl -sf "${BASE_URL}/models" >/dev/null 2>&1; do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "ERROR: vLLM exited during startup. Check the log above for the model load failure." >&2
    exit 1
  fi
  if (( SECONDS > deadline )); then
    echo "ERROR: vLLM did not become ready within ${PS_STARTUP_TIMEOUT}s." >&2
    echo "First load compiles HPU graphs and can be slow; raise PS_STARTUP_TIMEOUT or set VLLM_SKIP_WARMUP=true." >&2
    exit 1
  fi
  sleep 10
done
echo "=== vLLM ready ==="

# --- Run PaperScraper -------------------------------------------------------
activate_env "$PS_ENV"

# Configure the text profile through environment overrides rather than
# ps_model_config. ps_model_config persists to ~/.config/.pscraperrc.json,
# which concurrent jobs would race over; these variables are job-local.
export PAPERSCRAPER_MODEL_PROVIDER=local
export PAPERSCRAPER_MODEL_NAME="$PS_MODEL"
export PAPERSCRAPER_MODEL_BASE_URL="$BASE_URL"
export PAPERSCRAPER_MODEL_CAPABILITIES=text
export PAPERSCRAPER_MODEL_INPUT_TOKEN_LIMIT="$PS_INPUT_TOKEN_LIMIT"

ps_model_status

scrape_args=(--mode "$PS_MODE" --output "$PS_IN_CSV")
[[ -n "$PS_COUNT" ]] && scrape_args+=(--count "$PS_COUNT")

echo "=== Scraping ==="
ps_scrape "$PS_DB" "$PS_RECIPE" "${scrape_args[@]}"

if [[ "$PS_RUN_STORE" == "1" ]]; then
  # ps_store runs LLM-driven unit conversion, so it needs the server too.
  # --assume-yes is required: without it ps_store waits on an interactive
  # confirmation and the job would sit idle until it hits the walltime.
  echo "=== Storing ==="
  ps_store "$PS_DB" "$PS_IN_CSV" "$PS_OUT_CSV" "$PS_RECIPE" --assume-yes
fi

ps_status "$PS_DB"

# ps_scrape catches per-paper exceptions, records them on the row, and carries
# on, so a completely broken run still exits 0 and reports "0 material rows".
# Surface the recorded errors explicitly or they go unnoticed.
echo "=== Recorded per-paper errors ==="
python - "$PS_DB" <<'PY'
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
python - "$PS_DB" "$PS_MODE" <<'PY'
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
    print(f'  sbatch --export=ALL,PS_DB={db_path} scrape_gaudi.sbatch')
else:
    print()
    print('Corpus fully scraped.')
PY

echo "=== Done ==="
