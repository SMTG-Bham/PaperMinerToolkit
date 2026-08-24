#!/bin/bash
# Long-lived vLLM server on Gaudi, for reusing one warm model across many runs.
#
# Prefer scrape_gaudi.sbatch for ordinary work: it keeps the server and the
# scrape in a single job with nothing left behind. Use this variant when model
# load and HPU graph warmup cost more than the scrapes themselves, or when you
# want to drive the same server from several client jobs or a notebook.
#
# Submit from this directory, like everything else in this example. The log
# (vllm-gaudi-<jobid>.log) and the endpoint file land here beside the scripts:
#
#   cd examples/sol_gaudi
#   sbatch serve_gaudi.sbatch
#   cat vllm_endpoint.txt            # -> <node>:<port>
#
# Then, from a client job on a CPU partition or from your laptop through a
# tunnel, point PaperMiner at it (still from this directory, where the corpus
# is):
#
#   export PAPERSCRAPER_MODEL_PROVIDER=local
#   export PAPERSCRAPER_MODEL_NAME=Qwen/Qwen3-30B-A3B-Instruct-2507
#   export PAPERSCRAPER_MODEL_BASE_URL="http://$(cat vllm_endpoint.txt)/v1"
#   export PAPERSCRAPER_MODEL_CAPABILITIES=text
#   export PAPERSCRAPER_MODEL_INPUT_TOKEN_LIMIT=20000   # NOT equal to max-model-len
#   ps_scrape papers.db polymer --mode text --output scraped_polymer.csv
#
# The input limit must stay at least 11000 below PS_MAX_MODEL_LEN: PaperMiner
# always asks for a 10000-token completion and vLLM counts prompt + completion
# against --max-model-len. See scrape_gaudi.sbatch for the full explanation.

#SBATCH --job-name=ps-vllm-gaudi
#SBATCH -p gaudi
#SBATCH -q public
#SBATCH -N 1
#SBATCH -G 4
#SBATCH -c 18
# ASU's Gaudi examples all request -t 0-4. Eight hours is the point of running a
# long-lived server, but if the submit is rejected for exceeding the QOS limit,
# that is this line.
#SBATCH -t 0-08:00:00
#SBATCH -o vllm-gaudi-%j.log

set -euo pipefail

# Run from wherever the job was submitted — normally examples/sol_gaudi — so the
# endpoint file lands beside the scripts rather than in the scheduler's spool
# directory.
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Qwen3-30B-A3B-Instruct-2507 is the only Qwen3 model on Intel's validated list
# for Gaudi 2, at tensor parallel 4 or 8 - hence -G 4 above. The -Instruct-2507
# suffix is load-bearing: that variant is non-thinking by design, where a hybrid
# thinking model would spend PaperMiner's hardcoded 10000-token completion
# budget on reasoning before writing any JSON. 30.5B parameters at bf16 is ~61 GB
# of weights, ~15 GB per card across four.
PS_MODEL="${PS_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"

# Must equal the number of cards in the -G header above. ASU's own multi-card
# example is -G 8 -c 18 with --tensor-parallel-size 8: the core count does not
# scale with the cards, only -G and this do.
PS_TENSOR_PARALLEL_SIZE="${PS_TENSOR_PARALLEL_SIZE:-4}"

# ASU's own Gaudi examples serve at --max-model-len 8192, which is unusable here:
# a single 10000-token completion does not fit inside it, so every request would
# 400. Hence 32768, paid for with the memory knobs below.
PS_MAX_MODEL_LEN="${PS_MAX_MODEL_LEN:-32768}"

# Gaudi-specific launch flags, taken from https://docs.rc.asu.edu/vllm/. None of
# them are vLLM defaults and all three matter on HPU: 128 is the block size the
# Gaudi backend is tuned for, and holding utilisation and concurrent sequences
# down leaves device memory free for HPU graph capture during warmup. Lower
# PS_GPU_MEMORY_UTILIZATION first if warmup dies on an allocation failure.
PS_BLOCK_SIZE="${PS_BLOCK_SIZE:-128}"
PS_GPU_MEMORY_UTILIZATION="${PS_GPU_MEMORY_UTILIZATION:-0.80}"
PS_MAX_NUM_SEQS="${PS_MAX_NUM_SEQS:-16}"

PS_VLLM_ENV="${PS_VLLM_ENV:-gaudi-pytorch-vllm}"
# Beside the scripts, not in $HOME, so one directory holds everything this
# example produces. /home and /scratch are both shared across Sol, so a client
# job on another node reads it either way. Set ENDPOINT_FILE to move it.
ENDPOINT_FILE="${ENDPOINT_FILE:-$PWD/vllm_endpoint.txt}"

# Conda environments, unlike venvs, keep ~/.local/lib/pythonX.Y/site-packages on
# sys.path, ahead of the environment's own. One `pip install torch` made there
# with nothing activated shadows Habana's patched build for every later job, and
# the only symptom is an AssertionError deep inside vllm_gaudi about a
# compile-time/run-time PyTorch mismatch. Ignore user site for the whole job.
export PYTHONNOUSERSITE=1
export PT_HPU_LAZY_MODE=1
export HF_HOME="${HF_HOME:-/scratch/$USER/hf}"
mkdir -p "$HF_HOME"

# Required for multi-card tensor parallelism: ASU's vLLM page states outright
# that collective communication across HPUs needs it. Harmless on a single card,
# so it is set unconditionally rather than tied to the tensor parallel size.
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true

echo "=== Job ${SLURM_JOB_ID:-interactive} on $(hostname) ==="
echo "working directory: $PWD"
echo "model=$PS_MODEL tp=$PS_TENSOR_PARALLEL_SIZE max_model_len=$PS_MAX_MODEL_LEN"
hl-smi || echo "WARNING: hl-smi unavailable - are we really on a Gaudi node?"

# vLLM shards across exactly --tensor-parallel-size devices, so a mismatch with
# the -G allocation either dies during device discovery or hangs on the first
# collective, minutes into the model load. SLURM_GPUS_ON_NODE is the allocation
# as the scheduler granted it; hl-smi is not consulted, because on a shared node
# it can report cards this job was not given.
if [[ -n "${SLURM_GPUS_ON_NODE:-}" && "$SLURM_GPUS_ON_NODE" =~ ^[0-9]+$ ]] \
   && (( PS_TENSOR_PARALLEL_SIZE != SLURM_GPUS_ON_NODE )); then
  echo "ERROR: PS_TENSOR_PARALLEL_SIZE ($PS_TENSOR_PARALLEL_SIZE) does not match the" >&2
  echo "       $SLURM_GPUS_ON_NODE card(s) this job was allocated." >&2
  echo "       Change the -G line in this script, or set both at submit time:" >&2
  echo "         sbatch -G $PS_TENSOR_PARALLEL_SIZE serve_gaudi.sbatch" >&2
  exit 1
fi

# Module and conda activation scripts routinely reference unset variables, which
# would abort the job under `set -u`. Relax it around them only.
set +u
module load mamba/latest
source activate "$PS_VLLM_ENV"
set -u

# PYTHONNOUSERSITE covers ~/.local, but PYTHONPATH or a stray .pth file shadows
# torch just as well, and the mismatch only surfaces once vllm_gaudi imports
# habana_frameworks — minutes into a cold start. Check it now instead.
echo "=== Checking the vLLM environment ==="
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
  --tensor-parallel-size "$PS_TENSOR_PARALLEL_SIZE" \
  --max-model-len "$PS_MAX_MODEL_LEN" \
  --block-size "$PS_BLOCK_SIZE" \
  --gpu-memory-utilization "$PS_GPU_MEMORY_UTILIZATION" \
  --max-num-seqs "$PS_MAX_NUM_SEQS"
