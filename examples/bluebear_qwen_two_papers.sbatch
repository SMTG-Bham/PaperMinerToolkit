#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --qos=bbgpu
#SBATCH --account=scanlodo-sse-workflows
#SBATCH --time=1:00:0
#SBATCH --gres=gpu:a100_80:1

set -euo pipefail

module purge
module load bluebear

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
  echo "Could not initialise Conda. Make sure conda is available on BlueBEAR." >&2
  exit 1
fi

conda activate scraper

export PAPERSCRAPER_MODEL_HOST="${PAPERSCRAPER_MODEL_HOST:-127.0.0.1}"
export PAPERSCRAPER_MODEL_PORT="${PAPERSCRAPER_MODEL_PORT:-8000}"
export PAPERSCRAPER_MODEL_BASE_URL="${PAPERSCRAPER_MODEL_BASE_URL:-http://${PAPERSCRAPER_MODEL_HOST}:${PAPERSCRAPER_MODEL_PORT}/v1}"
export PAPERSCRAPER_MODEL_PROVIDER="${PAPERSCRAPER_MODEL_PROVIDER:-local}"
export PAPERSCRAPER_MODEL_NAME="${PAPERSCRAPER_MODEL_NAME:-Qwen/Qwen3-30B-A3B-FP8}"
export PAPERSCRAPER_MODEL_CAPABILITIES="${PAPERSCRAPER_MODEL_CAPABILITIES:-text}"
export PAPERSCRAPER_START_MODEL_SERVER="${PAPERSCRAPER_START_MODEL_SERVER:-1}"

echo "Using model endpoint: ${PAPERSCRAPER_MODEL_BASE_URL}"

if [[ "${PAPERSCRAPER_START_MODEL_SERVER}" == "1" ]]; then
  echo "Starting vLLM local model server for ${PAPERSCRAPER_MODEL_NAME}"
  python -m vllm.entrypoints.openai.api_server \
    --model "${PAPERSCRAPER_MODEL_NAME}" \
    --host "${PAPERSCRAPER_MODEL_HOST}" \
    --port "${PAPERSCRAPER_MODEL_PORT}" \
    ${VLLM_EXTRA_ARGS:-} &
  VLLM_PID=$!
  trap 'kill ${VLLM_PID} 2>/dev/null || true' EXIT

  python - <<'PY'
import os
import time
import urllib.request

base_url = os.environ['PAPERSCRAPER_MODEL_BASE_URL']
url = f'{base_url}/models'
last_error = None
for _ in range(180):
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            if response.status < 500:
                print(f'Model server is ready at {base_url}', flush=True)
                raise SystemExit(0)
    except Exception as exc:
        last_error = exc
    time.sleep(5)
raise SystemExit(f'Model server did not become ready at {base_url}: {last_error}')
PY
else
  echo "Skipping model server startup because PAPERSCRAPER_START_MODEL_SERVER=${PAPERSCRAPER_START_MODEL_SERVER}"
fi

if [[ -z "${ELSEVIER_API_KEY:-}" ]]; then
  echo "Configure the Elsevier API key before running, e.g. with ps_elsevier_key or your settings file." >&2
  exit 1
fi

python examples/bluebear_qwen_two_papers.py \
  --query "Li2NH AIMD solid electrolyte" \
  --model "${PAPERSCRAPER_MODEL_NAME}" \
  --base-url "${PAPERSCRAPER_MODEL_BASE_URL}" \
  --download-format both \
  --mode text
