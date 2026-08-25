#!/bin/bash
# Build the PaperMinerToolkit environment on Sol, ready for the Gaudi example.
#
# Submit this once, before anything else, from this directory:
#
#   cd examples/sol_gaudi
#   sbatch install.sbatch
#
# Every script in this example is submitted from here and writes back here, so
# the log lands beside the scripts:
#
#   tail -f ps-install-<jobid>.log
#
# The job itself runs from the repository root — the install needs
# build_tools/environment.yml and an editable install of the checkout — which it
# locates by walking up from this directory.
#
# It runs as a batch job because ASU asks that environments not be built on a
# login node. No accelerator is requested — this only installs an HTTP client.
#
# Options:
#   PMT_ENV=paperminertoolkit            name of the environment to create
#   PMT_FORCE=0                     reuse an existing environment instead of rebuilding
#   PMT_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507   model whose tokenizer to pre-cache
#   PMT_PREFETCH_WEIGHTS=1          also download the model weights (~61 GB for the default)
#
# When it finishes, from this same directory:
#
#   sbatch fetch_corpus.sbatch
#   sbatch scrape_gaudi.sbatch

#SBATCH --job-name=ps-install
#SBATCH -p htc
#SBATCH -q public
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 0-02:00:00
#SBATCH -o ps-install-%j.log

set -euo pipefail

PMT_ENV="${PMT_ENV:-paperminertoolkit}"
PMT_FORCE="${PMT_FORCE:-1}"
# Keep in step with the default in scrape_gaudi.sbatch: this only pre-caches
# the tokenizer that sizes text chunks, and a tokenizer for the wrong model is
# no better than none, since the lookup at scrape time misses and falls back to
# a character estimate.
PMT_MODEL="${PMT_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
PMT_PREFETCH_WEIGHTS="${PMT_PREFETCH_WEIGHTS:-0}"

# Never touch ASU's shared vLLM environment. It is Python 3.10 (below this
# package's floor) and pip would replace Habana's patched PyTorch with a stock
# wheel, breaking HPU support for everyone else on the cluster.
if [[ "$PMT_ENV" == "gaudi-pytorch-vllm" ]]; then
  echo "ERROR: refusing to install into the shared vLLM environment." >&2
  exit 1
fi

# Find the repository root by walking up from the submit directory. Unlike the
# other scripts here this one has to run from the root, for environment.yml and
# the editable install, but it is still submitted from examples/sol_gaudi like
# everything else. Slurm copies the batch script into its spool directory before
# running it, so $0 and BASH_SOURCE point at the copy and are no use for locating
# the repository — the submit directory is the only reliable anchor.
find_repo_root() {
  local dir="$1"
  while :; do
    if [[ -f "$dir/pyproject.toml" && -f "$dir/build_tools/environment.yml" ]]; then
      printf '%s\n' "$dir"
      return 0
    fi
    [[ "$dir" == "/" ]] && return 1
    dir="$(dirname "$dir")"
  done
}

submit_dir="$(cd "${SLURM_SUBMIT_DIR:-$PWD}" && pwd -P)"
if ! REPO_ROOT="$(find_repo_root "$submit_dir")"; then
  echo "ERROR: no PaperMinerToolkit checkout found at or above $submit_dir." >&2
  echo "       Submit from the example directory inside the checkout:" >&2
  echo "         cd examples/sol_gaudi && sbatch install.sbatch" >&2
  exit 1
fi
cd "$REPO_ROOT"
echo "=== Submitted from: $submit_dir ==="
echo "=== Repository root: $REPO_ROOT ==="

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
echo "env=$PMT_ENV  HF_HOME=$HF_HOME"

# Module and conda activation scripts reference unset variables, which would
# abort the job under `set -u`. Relax it around them only.
set +u
module load mamba/latest
set -u

# --- 0. A rebuild does not clear user site ----------------------------------
# Conda puts ~/.local/lib/pythonX.Y/site-packages ahead of the environment's own
# on sys.path, so a wheel there survives any number of rebuilds and shadows them
# all. The exports above stop this job reading or writing user site, and the
# verify step below fails if torch resolves outside the environment — but say so
# now rather than twenty minutes into a build, and name what to remove.
shadowing=()
for site in "$HOME"/.local/lib/python3.*/site-packages; do
  [[ -d "$site" ]] || continue
  for pkg in torch torchvision torchaudio triton; do
    if [[ -e "$site/$pkg" ]]; then
      shadowing+=("$site/$pkg")
    fi
  done
done
if (( ${#shadowing[@]} )); then
  echo "WARNING: these will shadow every environment on this account, including"
  echo "         the shared Gaudi one, for any job that does not set"
  echo "         PYTHONNOUSERSITE=1:"
  printf '           %s\n' "${shadowing[@]}"
  echo "         Clear them by hand, with no environment active:"
  echo "           python3 -m pip uninstall -y torch torchvision torchaudio"
  echo "         This script does not touch ~/.local: it is shared by everything"
  echo "         you run on this account, not just PaperMinerToolkit."
fi

# --- 1. Create the environment, from scratch --------------------------------
# Rebuilding is the default. Reusing an environment preserves exactly the state
# this script exists to rule out: a torch wheel swapped by hand, a half-finished
# pip install, a dependency pinned by an older environment.yml. Set PMT_FORCE=0
# to reuse one instead.
#
# Only $PMT_ENV is touched. No other environment on the account is removed, and
# the guard above keeps the shared vLLM environment out of reach entirely.
env_exists() {
  mamba env list | awk '{print $1}' | grep -Fxq "$1"
}

if env_exists "$PMT_ENV"; then
  if [[ "$PMT_FORCE" == "0" ]]; then
    echo "=== Reusing existing environment $PMT_ENV (PMT_FORCE=1 to rebuild) ==="
  else
    # $NF is the prefix column of `mamba env list`.
    PMT_PREFIX="$(mamba env list | awk -v name="$PMT_ENV" '$1 == name {print $NF}')"

    # Never remove an environment belonging to someone else: ASU's shared envs
    # live under /packages and a typo in PMT_ENV must not reach them.
    if [[ -n "$PMT_PREFIX" && ! -O "$PMT_PREFIX" ]]; then
      echo "ERROR: $PMT_ENV lives at $PMT_PREFIX, which you do not own." >&2
      echo "       Refusing to remove it. Pick a different PMT_ENV." >&2
      exit 1
    fi

    echo "=== Removing existing environment $PMT_ENV (${PMT_PREFIX:-prefix unknown}) ==="
    mamba env remove -n "$PMT_ENV" -y

    # mamba leaves the prefix behind when it holds files it did not install, and
    # `env create` then refuses with "prefix already exists". Clear the remains,
    # but only ever a path mamba itself reported for this environment.
    case "$PMT_PREFIX" in
      ''|'/'|"$HOME")
        ;;
      *)
        if [[ -d "$PMT_PREFIX" ]]; then
          echo "=== Clearing leftover prefix $PMT_PREFIX ==="
          rm -rf "$PMT_PREFIX"
        fi
        ;;
    esac
  fi
fi

if ! env_exists "$PMT_ENV"; then
  echo "=== Creating $PMT_ENV ==="
  mamba env create -f build_tools/environment.yml -n "$PMT_ENV"
fi

set +u
source activate "$PMT_ENV"
set -u
echo "python: $(python --version) at $(command -v python)"

# --- 2. CPU-only torch, before anything can pull the CUDA build --------------
# headroom-ai[ml] requires torch, which resolves to the CUDA build on x86_64
# Linux and drags in ~15 nvidia-* wheels plus triton — several GB that are dead
# weight on a Gaudi node, and confusing to have beside Habana's runtime.
# PaperMinerToolkit only touches torch through a lazy transformers import used for
# token counting, so the CPU build satisfies it completely. Installing it first
# means pip sees the constraint as already met and never considers a CUDA wheel.
echo "=== Installing CPU-only torch ==="
pip install --index-url https://download.pytorch.org/whl/cpu torch

echo "=== Installing PaperMinerToolkit ==="
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
             "A CUDA wheel was pulled in; check environment.yml and rerun.")
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

command -v pmt >/dev/null || { echo "ERROR: pmt not on PATH" >&2; exit 1; }
for command_name in search download scrape store status reset; do
  pmt "$command_name" --help >/dev/null
done
for group_name in corpus filter topics import config; do
  pmt "$group_name" --help >/dev/null
done
echo "pmt command hierarchy: OK"

echo "=== Test suite ==="
# Needs no credentials and no accelerator: the network and slow markers are
# excluded by pyproject.toml's default addopts.
pytest -q

# --- 4. Pre-cache the tokenizer --------------------------------------------
# For a local provider PaperMinerToolkit sizes its text chunks with
# transformers.AutoTokenizer.from_pretrained(<model name>), and silently falls
# back to a len/3 character estimate if that lookup fails. Caching it here, on a
# node with reliable outbound access, keeps the Gaudi job off the network.
echo "=== Pre-caching tokenizer for $PMT_MODEL ==="
if python -c "
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained('$PMT_MODEL')
print('tokenizer cached')
"; then
  echo "tokenizer: OK"
else
  echo "WARNING: could not cache the tokenizer for $PMT_MODEL."
  echo "         The scrape will still run but will size chunks by character"
  echo "         estimate. Check outbound access and the model name."
fi

if [[ "$PMT_PREFETCH_WEIGHTS" == "1" ]]; then
  # Downloading weights here rather than in the Gaudi job keeps tens of GB of
  # transfer out of a four-hour accelerator allocation - ~61 GB for the default
  # 30B model, which is worth doing once before the first scrape. Into the cache, not
  # --local-dir: a path-style --model would break the tokenizer lookup above.
  echo "=== Pre-fetching weights for $PMT_MODEL ==="
  hf download "$PMT_MODEL"
fi

echo
echo "=== Done. Environment '$PMT_ENV' is ready. ==="
echo "Next, from $submit_dir:"
echo "  sbatch fetch_corpus.sbatch"
echo "  sbatch scrape_gaudi.sbatch"
