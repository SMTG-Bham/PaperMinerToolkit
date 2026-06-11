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

if [[ -z "${PAPERSCRAPER_MODEL_BASE_URL:-}" ]]; then
  echo "PAPERSCRAPER_MODEL_BASE_URL must point at the OpenAI-compatible Qwen endpoint." >&2
  exit 1
fi

if [[ -z "${ELSEVIER_API_KEY:-}" ]]; then
  echo "Configure the Elsevier API key before running, e.g. with ps_elsevier_key or your settings file." >&2
  exit 1
fi

python examples/bluebear_qwen_two_papers.py \
  --query "Li2NH AIMD solid electrolyte" \
  --model "Qwen/Qwen3.6-35B-A3B-FP8" \
  --download-format both \
  --content both
