#!/bin/bash
# Submit one scrape_model.sbatch job per model in models.tsv.
#
#   cd examples/sol_gaudi_model_comparison
#   ./submit_comparison.sh --corpus /path/to/papers.db --count 5     # smoke test
#   ./submit_comparison.sh --corpus /path/to/papers.db               # full run
#   ./submit_comparison.sh --corpus /path/to/papers.db --only qwen25-72b
#
# Options:
#   --corpus PATH     corpus to compare on (required; copied per model on first run)
#   --only L[,L...]   submit only these labels from models.tsv
#   --count N         pmt scrape --count N, the same first N papers for every model
#   --time D-HH:MM:SS wall time per job (default 0-12:00:00; gaudi allows up to 7 days)
#   --models FILE     alternative lineup (default models.tsv)
#   --dependency SPEC passed to sbatch, e.g. afterok:<prefetch job id>
#   --dry-run         print the sbatch commands without submitting
#   --test-only       ask Slurm for a predicted start instead of submitting
#
# Each job gets as many Gaudi cards as its tensor-parallel size, a run
# directory runs/<label>/, and a log runs/<label>/scrape-<jobid>.log.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

CORPUS=""
ONLY=""
COUNT=""
WALL="0-12:00:00"
MODELS_FILE="models.tsv"
DRY_RUN=0
EXTRA=()

while (( $# )); do
  case "$1" in
    --corpus) CORPUS="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --count) COUNT="$2"; shift 2 ;;
    --time) WALL="$2"; shift 2 ;;
    --models) MODELS_FILE="$2"; shift 2 ;;
    --dependency) EXTRA+=(--dependency "$2"); shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --test-only) EXTRA+=(--test-only); shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$CORPUS" ]]; then
  echo "ERROR: --corpus PATH is required." >&2
  exit 2
fi
if [[ ! -f "$CORPUS" ]]; then
  echo "ERROR: corpus $CORPUS not found." >&2
  exit 1
fi
CORPUS="$(cd "$(dirname "$CORPUS")" && pwd -P)/$(basename "$CORPUS")"

submitted=0
while read -r label repo tp class; do
  if [[ -n "$ONLY" && ",$ONLY," != *",$label,"* ]]; then
    continue
  fi
  cmd=(sbatch "${EXTRA[@]}"
       --job-name="bgcmp-$label"
       -G "$tp"
       -t "$WALL"
       -o "runs/$label/scrape-%j.log"
       --export="ALL,PMT_LABEL=$label,PMT_MODEL=$repo,PMT_TENSOR_PARALLEL_SIZE=$tp,PMT_SOURCE_DB=$CORPUS,PMT_COUNT=$COUNT"
       scrape_model.sbatch)
  echo "# $label ($class): $repo on $tp card(s)"
  if (( DRY_RUN )); then
    printf '  %q' "${cmd[@]}"; echo
  else
    mkdir -p "runs/$label"
    "${cmd[@]}"
  fi
  submitted=$((submitted + 1))
done < <(grep -vE '^\s*(#|$)' "$MODELS_FILE")

if (( submitted == 0 )); then
  echo "ERROR: no models matched in $MODELS_FILE." >&2
  exit 1
fi
