#!/bin/bash
# Build a biodegradable polymer corpus before submitting any Gaudi work.
#
# Searching and downloading are network- and CPU-bound and never touch the
# model, so they must not run in the gaudi queue. Run this on a login node for
# a small corpus, or inside an htc job for a large one:
#
#   interactive -c 4 -t 0-2
#   ./examples/sol_gaudi/fetch_corpus.sh
#   ./examples/sol_gaudi/fetch_corpus.sh "polymer biodegradation OECD 310" bio.db 100
#
# The polymer and polymer_db recipes key off reported biodegradation tests, so
# queries naming the test standard (OECD 301, OECD 310, ASTM D6400, ISO 14855)
# tend to return papers with extractable results rather than review articles.
#
# Credentials come from the environment (see build_tools/sol_gaudi/README.md):
#   ELSEVIER_API_KEY, CORE_API_KEY, UNPAYWALL_EMAIL

set -euo pipefail

QUERY="${1:-biodegradable polymer OECD 301 biodegradation}"
DB="${2:-papers.db}"
COUNT="${3:-10}"
PS_ENV="${PS_ENV:-paperscraper}"

export HF_HOME="${HF_HOME:-/scratch/$USER/hf}"

# Module and conda activation scripts routinely reference unset variables, which
# would abort the run under `set -u`. Relax it around them only.
set +u
module load mamba/latest
source activate "$PS_ENV"
set -u

missing=()
[[ -n "${ELSEVIER_API_KEY:-}" ]] || missing+=(ELSEVIER_API_KEY)
[[ -n "${CORE_API_KEY:-}" ]]     || missing+=(CORE_API_KEY)
[[ -n "${UNPAYWALL_EMAIL:-}" ]]  || missing+=(UNPAYWALL_EMAIL)
if (( ${#missing[@]} )); then
  echo "WARNING: unset credentials: ${missing[*]}"
  echo "Those sources will be skipped or will return nothing."
fi

echo "=== Searching: $QUERY ==="
ps_search "$QUERY" "$DB" --source all --count "$COUNT"

echo "=== Downloading ==="
ps_download "$DB" --format both --source all

echo "=== Corpus ==="
ps_corpus_stats "$DB"

echo
echo "Corpus ready at $DB. Now submit the scrape (defaults to the polymer recipe):"
echo "  sbatch --export=ALL,PS_DB=$DB examples/sol_gaudi/scrape_gaudi.sbatch"
echo
echo "Start with one paper to check the pipeline end to end:"
echo "  sbatch --export=ALL,PS_DB=$DB,PS_COUNT=1 examples/sol_gaudi/scrape_gaudi.sbatch"
