#!/bin/bash
# Long-lived vLLM server on Gaudi, for reusing one warm model across many runs.
#
# Prefer scrape_gaudi.sbatch for ordinary work: it keeps the server and the
# scrape in a single job with nothing left behind. Use this variant when model
# load and HPU graph warmup cost more than the scrapes themselves, or when you
# want to drive the same server from several client jobs or a notebook.
#
#   sbatch examples/sol_gaudi/serve_gaudi.sbatch
#   cat ~/vllm_endpoint.txt          # -> <node>:<port>
#
# Then, from a client job on a CPU partition or from your laptop through a
# tunnel, point PaperScraper at it:
#
#   export PAPERSCRAPER_MODEL_PROVIDER=local
#   export PAPERSCRAPER_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
#   export PAPERSCRAPER_MODEL_BASE_URL="http://$(cat ~/vllm_endpoint.txt)/v1"
#   export PAPERSCRAPER_MODEL_CAPABILITIES=text
#   export PAPERSCRAPER_MODEL_INPUT_TOKEN_LIMIT=20000   # NOT equal to max-model-len
#   ps_scrape papers.db polymer --mode text --output scraped_polymer.csv
#
# The input limit must stay at least 11000 below PS_MAX_MODEL_LEN: PaperScraper
# always asks for a 10000-token completion and vLLM counts prompt + completion
# against --max-model-len. See scrape_gaudi.sbatch for the full explanation.

#SBATCH --job-name=ps-vllm-gaudi
#SBATCH -p gaudi
#SBATCH -q public
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 18
#SBATCH -t 0-08:00:00
#SBATCH -o vllm-gaudi-%j.log

set -euo pipefail

PS_MODEL="${PS_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PS_MAX_MODEL_LEN="${PS_MAX_MODEL_LEN:-32768}"
PS_VLLM_ENV="${PS_VLLM_ENV:-gaudi-pytorch-vllm}"
ENDPOINT_FILE="${ENDPOINT_FILE:-$HOME/vllm_endpoint.txt}"

export PT_HPU_LAZY_MODE=1
export HF_HOME="${HF_HOME:-/scratch/$USER/hf}"
mkdir -p "$HF_HOME"

# Multi-card only. Request the cards with -G 8 and add --tensor-parallel-size 8
# to the launch below.
# export PT_HPU_ENABLE_LAZY_COLLECTIVES=true

echo "=== Job ${SLURM_JOB_ID:-interactive} on $(hostname) ==="
hl-smi || echo "WARNING: hl-smi unavailable - are we really on a Gaudi node?"

# Module and conda activation scripts routinely reference unset variables, which
# would abort the job under `set -u`. Relax it around them only.
set +u
module load mamba/latest
source activate "$PS_VLLM_ENV"
set -u

PORT="$(shuf -i 8000-9000 -n 1)"

# Bind to 0.0.0.0 so client jobs on other nodes can reach it. Sol is behind a
# firewall, but this port is still open to anyone on the cluster - do not put
# anything sensitive behind it.
echo "$(hostname):${PORT}" > "$ENDPOINT_FILE"
echo "=== Endpoint written to $ENDPOINT_FILE: $(cat "$ENDPOINT_FILE") ==="
trap 'rm -f "$ENDPOINT_FILE"' EXIT

exec python -m vllm.entrypoints.openai.api_server \
  --model "$PS_MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --max-model-len "$PS_MAX_MODEL_LEN"
