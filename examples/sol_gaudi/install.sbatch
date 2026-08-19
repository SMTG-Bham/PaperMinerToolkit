#!/bin/bash
# Build the PaperScraper environment on Sol, ready for the Gaudi example.
#
# Submit this once from the repository root, before anything else:
#
#   sbatch examples/sol_gaudi/install.sbatch
#   tail -f ps-install-<jobid>.log
#
# It runs as a batch job because ASU asks that environments not be built on a
# login node. No accelerator is requested — this only installs an HTTP client.
#
# Options:
#   PS_ENV=paperscraper            name of the environment to create
#   PS_FORCE=1                     delete and rebuild an existing environment
#   PS_MODEL=Qwen/Qwen2.5-7B-Instruct   model whose tokenizer to pre-cache
#   PS_PREFETCH_WEIGHTS=1          also download the model weights (tens of GB)
#
# When it finishes:
#
#   sbatch examples/sol_gaudi/fetch_corpus.sbatch
#   sbatch examples/sol_gaudi/scrape_gaudi.sbatch

#SBATCH --job-name=ps-install
#SBATCH -p htc
#SBATCH -q public
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 0-02:00:00
#SBATCH -o ps-install-%j.log

set -euo pipefail

PS_ENV="${PS_ENV:-paperscraper}"
PS_FORCE="${PS_FORCE:-0}"
PS_MODEL="${PS_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PS_PREFETCH_WEIGHTS="${PS_PREFETCH_WEIGHTS:-0}"

# Never touch ASU's shared vLLM environment. It is Python 3.10 (below this
# package's floor) and pip would replace Habana's patched PyTorch with a stock
# wheel, breaking HPU support for everyone else on the cluster.
if [[ "$PS_ENV" == "gaudi-pytorch-vllm" ]]; then
  echo "ERROR: refusing to install into the shared vLLM environment." >&2
  exit 1
fi

cd "${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f pyproject.toml || ! -f build_tools/environment.yml ]]; then
  echo "ERROR: submit this from the repository root (no pyproject.toml here)." >&2
  exit 1
fi

# Conda environments, unlike venvs, keep ~/.local/lib/pythonX.Y/site-packages on
# sys.path, ahead of the environment's own — so a wheel landing there shadows
# this environment and ASU's shared Gaudi one for every later job. Neither read
# user site nor let pip write to it.
export PYTHONNOUSERSITE=1
export PIP_USER=0

# Keep caches off /home, which is capped at 100 GiB.
export HF_HOME="${HF_HOME:-/scratch/$USER/hf}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/scratch/$USER/pip}"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR"

echo "=== Job ${SLURM_JOB_ID:-interactive} on $(hostname) ==="
echo "env=$PS_ENV  HF_HOME=$HF_HOME"

# Module and conda activation scripts reference unset variables, which would
# abort the job under `set -u`. Relax it around them only.
set +u
module load mamba/latest
set -u

# --- 1. Create the environment ----------------------------------------------
if mamba env list | awk '{print $1}' | grep -Fxq "$PS_ENV"; then
  if [[ "$PS_FORCE" == "1" ]]; then
    echo "=== Removing existing environment $PS_ENV ==="
    mamba env remove -n "$PS_ENV" -y
    echo "=== Creating $PS_ENV ==="
    mamba env create -f build_tools/environment.yml -n "$PS_ENV"
  else
    echo "=== Environment $PS_ENV already exists; reusing it (PS_FORCE=1 to rebuild) ==="
  fi
else
  echo "=== Creating $PS_ENV ==="
  mamba env create -f build_tools/environment.yml -n "$PS_ENV"
fi

set +u
source activate "$PS_ENV"
set -u
echo "python: $(python --version) at $(command -v python)"

# --- 2. CPU-only torch, before anything can pull the CUDA build --------------
# headroom-ai[ml] requires torch, which resolves to the CUDA build on x86_64
# Linux and drags in ~15 nvidia-* wheels plus triton — several GB that are dead
# weight on a Gaudi node, and confusing to have beside Habana's runtime.
# PaperScraper only touches torch through a lazy transformers import used for
# token counting, so the CPU build satisfies it completely. Installing it first
# means pip sees the constraint as already met and never considers a CUDA wheel.
echo "=== Installing CPU-only torch ==="
pip install --index-url https://download.pytorch.org/whl/cpu torch

echo "=== Installing PaperScraper ==="
pip install -e ".[test]"

# --- 3. Assert the install is actually CUDA-free ----------------------------
echo "=== Verifying ==="
python - <<'PY'
import pathlib
import sys
import torch
print(f"torch {torch.__version__} from {pathlib.Path(torch.__file__).parent}")
if not torch.__version__.endswith("+cpu"):
    sys.exit(f"ERROR: expected a +cpu torch build, got {torch.__version__}. "
             "A CUDA wheel was pulled in; rerun with PS_FORCE=1.")
prefix = pathlib.Path(sys.prefix).resolve()
if prefix not in pathlib.Path(torch.__file__).resolve().parents:
    sys.exit(f"ERROR: torch resolved outside {prefix}. Something on sys.path is "
             "shadowing the environment; remove it before running any job.")
PY

cuda_pkgs="$(pip list --format=freeze | grep -cE '^(nvidia-|triton)' || true)"
if [[ "$cuda_pkgs" != "0" ]]; then
  echo "ERROR: $cuda_pkgs CUDA packages were installed:" >&2
  pip list --format=freeze | grep -E '^(nvidia-|triton)' >&2
  exit 1
fi
echo "CUDA packages: 0"

for cmd in ps_search ps_download ps_scrape ps_store ps_status ps_model_status; do
  command -v "$cmd" >/dev/null || { echo "ERROR: $cmd not on PATH" >&2; exit 1; }
done
echo "console scripts: OK"

echo "=== Test suite ==="
# Needs no credentials and no accelerator: the network and slow markers are
# excluded by pyproject.toml's default addopts.
pytest -q

# --- 4. Pre-cache the tokenizer --------------------------------------------
# For a local provider PaperScraper sizes its text chunks with
# transformers.AutoTokenizer.from_pretrained(<model name>), and silently falls
# back to a len/3 character estimate if that lookup fails. Caching it here, on a
# node with reliable outbound access, keeps the Gaudi job off the network.
echo "=== Pre-caching tokenizer for $PS_MODEL ==="
if python -c "
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained('$PS_MODEL')
print('tokenizer cached')
"; then
  echo "tokenizer: OK"
else
  echo "WARNING: could not cache the tokenizer for $PS_MODEL."
  echo "         The scrape will still run but will size chunks by character"
  echo "         estimate. Check outbound access and the model name."
fi

if [[ "$PS_PREFETCH_WEIGHTS" == "1" ]]; then
  # Downloading weights here rather than in the Gaudi job keeps tens of GB of
  # transfer out of a four-hour accelerator allocation. Into the cache, not
  # --local-dir: a path-style --model would break the tokenizer lookup above.
  echo "=== Pre-fetching weights for $PS_MODEL ==="
  hf download "$PS_MODEL"
fi

echo
echo "=== Done. Environment '$PS_ENV' is ready. ==="
echo "Next:"
echo "  sbatch examples/sol_gaudi/fetch_corpus.sbatch"
echo "  sbatch examples/sol_gaudi/scrape_gaudi.sbatch"
