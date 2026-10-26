#!/bin/bash
# Download the comparison models' weights into the scratch Hugging Face cache,
# off the Gaudi queue, so no accelerator allocation is spent on transfers.
#
# Submit from this directory, once the paperminertoolkit environment exists
# (see README.md, step 1). One array task per line of models.tsv:
#
#   cd examples/sol_gaudi_model_comparison
#   sbatch --array=0-3 prefetch_weights.sbatch
#
# Without --array the single job downloads every model in turn, which can
# outrun the four-hour htc limit for the full lineup. Re-running is cheap: the
# cache is checked file by file and only missing pieces are fetched.
#
# Gated repositories need HF_TOKEN. Put it in ~/.paperminertoolkit_env (mode
# 600) and the job sources that file; nothing is printed.

#SBATCH --job-name=bgcmp-prefetch
#SBATCH -p htc
#SBATCH -q public
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 0-04:00:00
#SBATCH -o prefetch-%A-%a.log

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
MODELS_FILE="${PMT_MODELS_FILE:-models.tsv}"
PMT_ENV="${PMT_ENV:-paperminertoolkit}"

export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-/scratch/$USER/hf}"
mkdir -p "$HF_HOME"

if [[ -r "$HOME/.paperminertoolkit_env" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/.paperminertoolkit_env"
fi

set +u
module load mamba/latest
source activate "$PMT_ENV"
set -u

# Lines of models.tsv, comments and blanks dropped: "label repo tp class".
mapfile -t rows < <(grep -vE '^\s*(#|$)' "$MODELS_FILE")
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  if (( SLURM_ARRAY_TASK_ID >= ${#rows[@]} )); then
    echo "ERROR: array index $SLURM_ARRAY_TASK_ID but $MODELS_FILE has ${#rows[@]} models." >&2
    exit 1
  fi
  rows=("${rows[$SLURM_ARRAY_TASK_ID]}")
fi

for row in "${rows[@]}"; do
  read -r label repo _tp _class <<<"$row"
  echo "=== $label: $repo -> $HF_HOME ($(date -u +%FT%TZ)) ==="
  # Only the safetensors shards and the small config files vLLM reads. Some
  # repositories also carry .bin or "original/" checkpoints that would double
  # the transfer for nothing.
  python - "$repo" <<'PY'
import sys
from huggingface_hub import snapshot_download

path = snapshot_download(
    sys.argv[1],
    allow_patterns=['*.safetensors', '*.json', '*.txt', '*.model', '*.tiktoken', '*.py', '*.jinja'],
    ignore_patterns=['original/*', 'consolidated*'],
)
print(f'cached at {path}')
PY
done
echo "=== Done ($(date -u +%FT%TZ)) ==="
