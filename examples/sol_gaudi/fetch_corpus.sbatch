#!/bin/bash
# Submit the biodegradable polymer corpus build as a batch job.
#
# This is a wrapper around fetch_corpus.sh for when the corpus is too large to
# babysit in an interactive session. Search and download are rate-limit bound
# rather than compute bound, so this asks for few cores and a lot of wall clock.
#
#   sbatch examples/sol_gaudi/fetch_corpus.sbatch
#   sbatch --export=ALL,PS_COUNT=500 examples/sol_gaudi/fetch_corpus.sbatch   # bound it
#   sbatch --export=ALL,PS_QUERY="polymer biodegradation OECD 310",PS_DB=bio.db \
#          examples/sol_gaudi/fetch_corpus.sbatch
#
# Credentials: a batch job does not reliably inherit your shell environment, so
# keep the keys in a file only you can read and this script will source it:
#
#   printf 'export ELSEVIER_API_KEY=...\nexport CORE_API_KEY=...\nexport UNPAYWALL_EMAIL=you@asu.edu\n' > ~/.paperscraper_env
#   chmod 600 ~/.paperscraper_env
#
# When the job finishes, submit the scrape:
#
#   sbatch examples/sol_gaudi/scrape_gaudi.sbatch

#SBATCH --job-name=ps-fetch-corpus
#SBATCH -p htc
#SBATCH -q public
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 0-04:00:00
#SBATCH -o ps-fetch-%j.log

# Deliberately no -G. Search and download never call the model, and a Gaudi card
# would sit idle for the whole job waiting on Crossref and Unpaywall.
#
# htc caps wall time at 4 hours but runs without interruption.
#
# Mind that against the unbounded PS_COUNT below. Search is cheap and re-running
# it is nearly free, because ps_search upserts on DOI. Download is not: ps_download
# walks every row in the corpus and has no skip for assets it already has, so a
# job killed at the wall clock re-downloads from the start on the next submit.
# If the corpus is big enough for that to bite, either bound PS_COUNT so the job
# finishes inside four hours, or move to a partition that allows longer:
#
#   #SBATCH -p general
#   #SBATCH -t 1-00:00:00

set -euo pipefail

PS_QUERY="${PS_QUERY:-biodegradable polymer OECD 301 biodegradation}"
PS_DB="${PS_DB:-papers.db}"
# Every paper the sources will give us, per source. See fetch_corpus.sh for what
# the number means; set it to something small while testing a new query.
PS_COUNT="${PS_COUNT:-1000000}"
PS_SECRETS="${PS_SECRETS:-$HOME/.paperscraper_env}"

# Run from wherever the job was submitted so papers.db lands next to the repo,
# not in the scheduler's spool directory.
cd "${SLURM_SUBMIT_DIR:-$PWD}"

PS_FETCH="${PS_FETCH:-examples/sol_gaudi/fetch_corpus.sh}"
if [[ ! -f "$PS_FETCH" ]]; then
  echo "ERROR: $PS_FETCH not found. Submit from the repository root, or set" >&2
  echo "       PS_FETCH to the path of fetch_corpus.sh." >&2
  exit 1
fi

if [[ -f "$PS_SECRETS" ]]; then
  echo "=== Sourcing credentials from $PS_SECRETS ==="
  set +u
  source "$PS_SECRETS"
  set -u
else
  echo "NOTE: no $PS_SECRETS found; relying on whatever the job inherited."
fi

echo "=== Job ${SLURM_JOB_ID:-interactive} on $(hostname) ==="
echo "query=$PS_QUERY db=$PS_DB count=$PS_COUNT"

# bash rather than direct execution, so a missing exec bit is not fatal.
exec bash "$PS_FETCH" "$PS_QUERY" "$PS_DB" "$PS_COUNT"
