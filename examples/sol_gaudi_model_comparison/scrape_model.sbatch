#!/bin/bash
# Scrape one copy of the corpus with one model, so several models can be
# compared on identical input.
#
# Do not submit this directly: submit_comparison.sh reads models.tsv and
# submits one of these jobs per model with the right card count, label and
# log path. The job is self-contained otherwise, and mirrors
# ../sol_gaudi/scrape_gaudi.sbatch: it starts vLLM from ASU's shared
# gaudi-pytorch-vllm environment on the allocated Gaudi cards, waits for it,
# runs pmt scrape from our own environment against 127.0.0.1, and stops the
# server on exit.
#
# Each model works in runs/<label>/ under this directory:
#   papers.db              its own copy of the corpus, staged on the first run,
#                          because scrape statuses live in the corpus and two
#                          models sharing one database would skip each other's
#                          papers
#   scraped_<recipe>.csv   the extracted records, appended across resubmissions
#   run_history.jsonl      one line per job: model, settings, versions, and
#                          timing split into server start-up and scrape phases
#   scrape-<jobid>.log     this log
#
# The scrape is resumable. A job that hits the wall clock loses only the paper
# in flight; resubmitting the same label continues where it stopped.

#SBATCH -p gaudi
#SBATCH -q public
#SBATCH -N 1
#SBATCH -c 18
#SBATCH -t 0-12:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
EXAMPLE_DIR="$PWD"
GAUDI_DIR="$EXAMPLE_DIR/../sol_gaudi"

# --- Required from the submitter ---------------------------------------------
: "${PMT_LABEL:?set by submit_comparison.sh}"
: "${PMT_MODEL:?set by submit_comparison.sh}"
: "${PMT_TENSOR_PARALLEL_SIZE:?set by submit_comparison.sh}"
: "${PMT_SOURCE_DB:?set by submit_comparison.sh}"

# --- Knobs, overridable with --export ----------------------------------------
PMT_RECIPE="${PMT_RECIPE:-band_gap_validation}"
PMT_MODE="${PMT_MODE:-text}"
# Every model sees the papers in the same order, so a bounded smoke test
# (PMT_COUNT=5) selects the same five papers for each model.
PMT_ORDER="${PMT_ORDER:-paper-id}"
PMT_COUNT="${PMT_COUNT:-}"
PMT_SUPPLEMENTS="${PMT_SUPPLEMENTS:-1}"
# See scrape_gaudi.sbatch for why these three are not equal: pmt always asks
# for a 10000-token completion and vLLM counts prompt + completion against
# --max-model-len. 32768 fits every model in models.tsv without RoPE scaling.
PMT_MAX_MODEL_LEN="${PMT_MAX_MODEL_LEN:-32768}"
PMT_INPUT_TOKEN_LIMIT="${PMT_INPUT_TOKEN_LIMIT:-20000}"
PMT_OUTPUT_RESERVE="${PMT_OUTPUT_RESERVE:-11000}"
PMT_BLOCK_SIZE="${PMT_BLOCK_SIZE:-128}"
PMT_GPU_MEMORY_UTILIZATION="${PMT_GPU_MEMORY_UTILIZATION:-0.80}"
PMT_MAX_NUM_SEQS="${PMT_MAX_NUM_SEQS:-16}"
PMT_STARTUP_TIMEOUT="${PMT_STARTUP_TIMEOUT:-3600}"
PMT_ENV="${PMT_ENV:-paperminertoolkit}"
PMT_VLLM_ENV="${PMT_VLLM_ENV:-gaudi-pytorch-vllm}"

RUN_DIR="$EXAMPLE_DIR/runs/$PMT_LABEL"
DB="$RUN_DIR/papers.db"
OUT_CSV="$RUN_DIR/scraped_${PMT_RECIPE}.csv"
HISTORY="$RUN_DIR/run_history.jsonl"
mkdir -p "$RUN_DIR"

export PYTHONNOUSERSITE=1
export PT_HPU_LAZY_MODE=1
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export HF_HOME="${HF_HOME:-/scratch/$USER/hf}"
mkdir -p "$HF_HOME"

STARTED="$(date -u +%FT%TZ)"
echo "=== Job ${SLURM_JOB_ID:-interactive} on $(hostname), $STARTED ==="
echo "label=$PMT_LABEL model=$PMT_MODEL tp=$PMT_TENSOR_PARALLEL_SIZE"
echo "recipe=$PMT_RECIPE mode=$PMT_MODE order=$PMT_ORDER count=${PMT_COUNT:-all}"
echo "max_model_len=$PMT_MAX_MODEL_LEN input_token_limit=$PMT_INPUT_TOKEN_LIMIT"
hl-smi || echo "WARNING: hl-smi unavailable - are we really on a Gaudi node?"

if (( PMT_MAX_MODEL_LEN < PMT_INPUT_TOKEN_LIMIT + PMT_OUTPUT_RESERVE )); then
  echo "ERROR: PMT_MAX_MODEL_LEN must be at least PMT_INPUT_TOKEN_LIMIT + $PMT_OUTPUT_RESERVE." >&2
  exit 1
fi
if [[ -n "${SLURM_GPUS_ON_NODE:-}" && "$SLURM_GPUS_ON_NODE" =~ ^[0-9]+$ ]] \
   && (( PMT_TENSOR_PARALLEL_SIZE != SLURM_GPUS_ON_NODE )); then
  echo "ERROR: PMT_TENSOR_PARALLEL_SIZE ($PMT_TENSOR_PARALLEL_SIZE) != allocated cards ($SLURM_GPUS_ON_NODE)." >&2
  exit 1
fi

activate_env() {
  set +u
  source activate "$1"
  set -u
}
set +u
module load mamba/latest
set -u

# --- Stage this model's copy of the corpus -----------------------------------
# The SQLite backup API gives a consistent copy even if the source has a WAL
# file, which a plain cp would not.
if [[ ! -f "$DB" ]]; then
  if [[ ! -f "$PMT_SOURCE_DB" ]]; then
    echo "ERROR: source corpus $PMT_SOURCE_DB not found." >&2
    exit 1
  fi
  echo "=== Staging $PMT_SOURCE_DB -> $DB ==="
  (
    activate_env "$PMT_ENV"
    python - "$PMT_SOURCE_DB" "$DB" <<'PY'
import sqlite3
import sys

source, target = sys.argv[1:3]
with sqlite3.connect(source) as src, sqlite3.connect(target) as dst:
    src.backup(dst)
PY
  )
fi

# --- Provenance for run_history.jsonl ----------------------------------------
COMMIT="$(git -C "$EXAMPLE_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
REVISION="$(cat "$HF_HOME/hub/models--${PMT_MODEL//\//--}/refs/main" 2>/dev/null || echo unknown)"
VLLM_VERSION="$(
  activate_env "$PMT_VLLM_ENV"
  python -c 'import importlib.metadata as m; print(m.version("vllm"))' 2>/dev/null || echo unknown
)"
PMT_VERSION="$(
  activate_env "$PMT_ENV"
  python -c 'import paperminertoolkit; print(paperminertoolkit.__version__)' 2>/dev/null || echo unknown
)"

record_run() {
  # One JSON line per job. summarize_comparison.py sums the elapsed times, so a
  # label resubmitted three times reports its total wall time.
  python3 - "$HISTORY" "$1" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

history, status = sys.argv[1:3]
env = os.environ
finished = datetime.now(timezone.utc)
started = datetime.strptime(env['STARTED'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
entry = {
    'label': env['PMT_LABEL'], 'model': env['PMT_MODEL'],
    'tensor_parallel': int(env['PMT_TENSOR_PARALLEL_SIZE']),
    'recipe': env['PMT_RECIPE'], 'mode': env['PMT_MODE'], 'order': env['PMT_ORDER'],
    'count': env.get('PMT_COUNT') or None,
    'max_model_len': int(env['PMT_MAX_MODEL_LEN']),
    'input_token_limit': int(env['PMT_INPUT_TOKEN_LIMIT']),
    'model_revision': env.get('REVISION'), 'vllm_version': env.get('VLLM_VERSION'),
    'paperminertoolkit_version': env.get('PMT_VERSION'), 'commit': env.get('COMMIT'),
    'job_id': env.get('SLURM_JOB_ID'), 'node': env.get('SLURMD_NODENAME'),
    'started': env['STARTED'], 'finished': finished.strftime('%Y-%m-%dT%H:%M:%SZ'),
    'elapsed_s': int((finished - started).total_seconds()), 'exit_status': int(status),
}
# The scrape phase starts once vLLM answers; before that the job is loading
# weights and compiling HPU graphs. Both halves are kept so the summary can
# report server start-up and per-paper throughput separately.
scrape_started = env.get('SCRAPE_STARTED')
if scrape_started:
    ready = datetime.strptime(scrape_started, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
    entry['scrape_started'] = scrape_started
    entry['server_startup_s'] = int((ready - started).total_seconds())
    entry['scrape_elapsed_s'] = int((finished - ready).total_seconds())
with open(history, 'a', encoding='utf-8') as handle:
    handle.write(json.dumps(entry) + '\n')
PY
}
export STARTED COMMIT REVISION VLLM_VERSION PMT_VERSION PMT_RECIPE PMT_MODE PMT_ORDER \
       PMT_MAX_MODEL_LEN PMT_INPUT_TOKEN_LIMIT PMT_LABEL PMT_MODEL PMT_TENSOR_PARALLEL_SIZE PMT_COUNT

VLLM_PID=""
cleanup() {
  local status=$?
  if [[ -n "$VLLM_PID" ]]; then
    echo "=== Stopping vLLM (pid $VLLM_PID) ==="
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
  record_run "$status"
}
trap cleanup EXIT

# --- Start vLLM ---------------------------------------------------------------
echo "=== Checking the vLLM environment ==="
(
  activate_env "$PMT_VLLM_ENV"
  python -u "$GAUDI_DIR/check_gaudi_environment.py"
)

PORT="$(shuf -i 8000-9000 -n 1)"
BASE_URL="http://127.0.0.1:${PORT}/v1"
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

echo "=== Waiting for vLLM on port $PORT (timeout ${PMT_STARTUP_TIMEOUT}s) ==="
deadline=$(( SECONDS + PMT_STARTUP_TIMEOUT ))
until curl -sf "${BASE_URL}/models" >/dev/null 2>&1; do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "ERROR: vLLM exited during startup; see the model load failure above." >&2
    exit 1
  fi
  if (( SECONDS > deadline )); then
    echo "ERROR: vLLM not ready within ${PMT_STARTUP_TIMEOUT}s; raise PMT_STARTUP_TIMEOUT." >&2
    exit 1
  fi
  sleep 10
done
SCRAPE_STARTED="$(date -u +%FT%TZ)"
export SCRAPE_STARTED
echo "=== vLLM ready ($SCRAPE_STARTED) ==="

# --- Scrape -------------------------------------------------------------------
activate_env "$PMT_ENV"
export PAPERMINERTOOLKIT_MODEL_PROVIDER=local
export PAPERMINERTOOLKIT_MODEL_NAME="$PMT_MODEL"
export PAPERMINERTOOLKIT_MODEL_BASE_URL="$BASE_URL"
export PAPERMINERTOOLKIT_MODEL_CAPABILITIES=text
export PAPERMINERTOOLKIT_MODEL_INPUT_TOKEN_LIMIT="$PMT_INPUT_TOKEN_LIMIT"
# Greedy decoding is the toolkit default; stated here so the comparison is
# explicitly deterministic per model.
export PAPERMINERTOOLKIT_MODEL_TEMPERATURE=0
pmt config status

scrape_args=(--mode "$PMT_MODE" --order "$PMT_ORDER" --output "$OUT_CSV")
[[ -n "$PMT_COUNT" ]] && scrape_args+=(--count "$PMT_COUNT")
[[ "$PMT_SUPPLEMENTS" == "1" ]] || scrape_args+=(--no-supplements)

echo "=== Scraping ($(date -u +%FT%TZ)) ==="
pmt scrape "$DB" "$PMT_RECIPE" "${scrape_args[@]}"
pmt status "$DB"

echo "=== Recorded per-paper errors ==="
python - "$DB" <<'PY'
import sqlite3
import sys

con = sqlite3.connect(sys.argv[1])
rows = con.execute(
    "SELECT substr(last_error, 1, 160), COUNT(*) FROM papers"
    " WHERE COALESCE(last_error, '') != '' GROUP BY 1 ORDER BY 2 DESC LIMIT 10"
).fetchall()
print('none' if not rows else '\n'.join(f'[{n}x] {err}' for err, n in rows))
PY

echo "=== Remaining work ==="
python - "$DB" "$PMT_MODE" "$PMT_LABEL" <<'PY'
import sqlite3
import sys

db_path, mode, label = sys.argv[1:4]
column = 'abstract_scrape_status' if mode == 'abstract' else 'text_scrape_status'
con = sqlite3.connect(db_path)
total = con.execute('SELECT COUNT(*) FROM papers').fetchone()[0]
done = con.execute(f"SELECT COUNT(*) FROM papers WHERE COALESCE({column}, '') = 'succeeded'").fetchone()[0]
failed = con.execute(f"SELECT COUNT(*) FROM papers WHERE COALESCE({column}, '') = 'failed'").fetchone()[0]
print(f'{column}: {done} succeeded, {failed} failed, {total - done - failed} untouched of {total}')
if total - done - failed:
    print(f'Continue this label with: ./submit_comparison.sh --only {label} ...')
else:
    print('Nothing left to scrape for this label.')
PY
echo "=== Done ($(date -u +%FT%TZ)) ==="
