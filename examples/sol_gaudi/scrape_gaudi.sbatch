#!/bin/bash
# Scrape biodegradable polymer data with a local vLLM server on one Gaudi 2 card.
#
# The job starts vLLM from ASU Research Computing's shared gaudi-pytorch-vllm
# environment, waits for it to serve, then runs ps_scrape from our own
# paperscraper environment against 127.0.0.1. Both live in the same job, so
# nothing is left running and no cross-node networking is involved.
#
# Submit from the directory holding papers.db:
#
#   sbatch examples/sol_gaudi/scrape_gaudi.sbatch
#
# Every setting below can be overridden at submit time, for example:
#
#   sbatch --export=ALL,PS_RECIPE=polymer_db,PS_COUNT=5 examples/sol_gaudi/scrape_gaudi.sbatch

#SBATCH --job-name=ps-scrape-gaudi
#SBATCH -p gaudi
#SBATCH -q public
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 18
#SBATCH -t 0-04:00:00
#SBATCH -o ps-gaudi-%j.log

set -euo pipefail

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
PS_MAX_MODEL_LEN="${PS_MAX_MODEL_LEN:-32768}"
PS_INPUT_TOKEN_LIMIT="${PS_INPUT_TOKEN_LIMIT:-20000}"
PS_OUTPUT_RESERVE="${PS_OUTPUT_RESERVE:-11000}"

PS_DB="${PS_DB:-papers.db}"
PS_MODE="${PS_MODE:-text}"
PS_COUNT="${PS_COUNT:-}"            # empty means every eligible paper

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
PS_ENV="${PS_ENV:-paperscraper}"    # our env, created per build_tools/sol_gaudi/README.md
PS_VLLM_ENV="${PS_VLLM_ENV:-gaudi-pytorch-vllm}"   # ASU's shared env - never install into it
PS_RUN_STORE="${PS_RUN_STORE:-1}"   # ps_store also calls the model, so do it while the server is up
PS_STARTUP_TIMEOUT="${PS_STARTUP_TIMEOUT:-3600}"

# --- Gaudi and cache environment -------------------------------------------
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
echo "model=$PS_MODEL max_model_len=$PS_MAX_MODEL_LEN mode=$PS_MODE recipe=$PS_RECIPE"
hl-smi || echo "WARNING: hl-smi unavailable - are we really on a Gaudi node?"

if [[ ! -f "$PS_DB" ]]; then
  echo "ERROR: $PS_DB not found. Build the corpus first with examples/sol_gaudi/fetch_corpus.sh" >&2
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

# --- Start vLLM -------------------------------------------------------------
# The subshell is load-bearing: activating our env later in the parent shell
# would otherwise clobber the interpreter the server is running under.
(
  activate_env "$PS_VLLM_ENV"
  exec python -m vllm.entrypoints.openai.api_server \
    --model "$PS_MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --max-model-len "$PS_MAX_MODEL_LEN"
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

echo "=== Done ==="
