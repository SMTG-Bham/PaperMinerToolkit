#!/bin/bash
# Build a biodegradable polymer corpus before submitting any Gaudi work.
#
# Searching and downloading are network- and CPU-bound and never touch the
# model, so they must not run in the gaudi queue. Run this on a login node for
# a small corpus, or submit fetch_corpus.sbatch for a large one:
#
#   interactive -c 4 -t 0-2
#   cd examples/sol_gaudi
#   ./fetch_corpus.sh
#   ./fetch_corpus.sh "polymer biodegradation OECD 310" bio.db 100
#
# A relative database path is taken as relative to this directory, wherever you
# happen to run the script from, so the corpus always lands beside the scripts
# where scrape_gaudi.sbatch looks for it. Pass an absolute path to put it
# somewhere else.
#
# The third argument caps results per source and defaults to everything the
# sources have. An unbounded query against a broad search term can return tens of
# thousands of papers and download for many hours - pass a number to bound it, or
# submit fetch_corpus.sbatch instead of running this interactively.
#
# The polymer and polymer_db recipes key off reported biodegradation tests, so
# queries naming the test standard (OECD 301, OECD 310, ASTM D6400, ISO 14855)
# tend to return papers with extractable results rather than review articles.
#
# Credentials come from the environment (see examples/sol_gaudi/README.md):
#   ELSEVIER_API_KEY, CORE_API_KEY, UNPAYWALL_EMAIL

set -euo pipefail

# Work from the directory holding this script, so the corpus ends up beside the
# sbatch scripts that consume it. This one is executed in place rather than
# copied into Slurm's spool directory, so BASH_SOURCE really does point at it.
cd "$(dirname "${BASH_SOURCE[0]}")"

QUERY="${1:-biodegradable polymer OECD 301 biodegradation}"
DB="${2:-papers.db}"

# Every paper the sources will give us. ps_search has no "unlimited" flag: each
# backend loops until it has COUNT records or the provider runs out, so a number
# larger than any real result set is how you ask for everything. Scopus stops at
# its own total, CORE and OpenAlex stop when a short page comes back.
#
# COUNT is per source, not in total. --source all queries Scopus, CORE and
# OpenAlex with the same number and merges the results on DOI, so the corpus
# ends up somewhere between the largest single source and the sum of all three.
COUNT="${3:-1000000}"
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

# $DB is absolute when fetch_corpus.sbatch resolved it against the submit
# directory, and relative when it came straight off the command line.
case "$DB" in
  /*) DB_DISPLAY="$DB" ;;
  *)  DB_DISPLAY="$PWD/$DB" ;;
esac

echo
echo "Corpus ready at $DB_DISPLAY. Now submit the scrape from $PWD"
echo "(it defaults to the polymer recipe):"
echo "  sbatch --export=ALL,PS_DB=$DB scrape_gaudi.sbatch"
echo
echo "Start with one paper to check the pipeline end to end:"
echo "  sbatch --export=ALL,PS_DB=$DB,PS_COUNT=1 scrape_gaudi.sbatch"
