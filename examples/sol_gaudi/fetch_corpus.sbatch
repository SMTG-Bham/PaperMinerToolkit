#!/bin/bash
# Submit the biodegradable polymer corpus build as a batch job.
#
# This is a wrapper around fetch_corpus.sh for when the corpus is too large to
# babysit in an interactive session. Search and download are rate-limit bound
# rather than compute bound, so this asks for few cores and a lot of wall clock.
#
# Submit from this directory, like everything else in this example:
#
#   cd examples/sol_gaudi
#   sbatch fetch_corpus.sbatch
#   sbatch --export=ALL,PMT_COUNT=500 fetch_corpus.sbatch   # bound it
#   sbatch --export=ALL,PMT_QUERY="polymer biodegradation OECD 310",PMT_DB=bio.db \
#          fetch_corpus.sbatch
#
# The log (ps-fetch-<jobid>.log) and the corpus land beside the scripts, where
# scrape_gaudi.sbatch looks for them.
#
# Credentials: a batch job does not reliably inherit your shell environment, so
# keep the keys in a file only you can read and this script will source it:
#
#   printf 'export ELSEVIER_API_KEY=...\nexport CORE_API_KEY=...\nexport UNPAYWALL_EMAIL=you@asu.edu\nexport OPENALEX_API_KEY=...\n' > ~/.paperminertoolkit_env
#   chmod 600 ~/.paperminertoolkit_env
#
# When the job finishes, submit the scrape from the same directory:
#
#   sbatch scrape_gaudi.sbatch

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
# Mind that against the unbounded PMT_COUNT below. Search is cheap and re-running
# it is nearly free, because pmt search upserts on DOI. Download is not: pmt download
# walks every row in the corpus and has no skip for assets it already has, so a
# job killed at the wall clock re-downloads from the start on the next submit.
# If the corpus is big enough for that to bite, either bound PMT_COUNT so the job
# finishes inside four hours, or move to a partition that allows longer:
#
#   #SBATCH -p general
#   #SBATCH -t 1-00:00:00

set -euo pipefail

PMT_QUERY="${PMT_QUERY:-biodegradable polymer OECD 301 biodegradation}"
PMT_DB="${PMT_DB:-papers.db}"
# Every paper the sources will give us, per source. See fetch_corpus.sh for what
# the number means; set it to something small while testing a new query.
PMT_COUNT="${PMT_COUNT:-1000000}"
PMT_SECRETS="${PMT_SECRETS:-$HOME/.paperminertoolkit_env}"

# Run from wherever the job was submitted — normally examples/sol_gaudi — so the
# corpus lands beside the scripts rather than in the scheduler's spool directory.
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# fetch_corpus.sh anchors a relative database to its own directory. Resolve
# PMT_DB here so this job's working directory wins even if PMT_FETCH points at a
# copy of the script kept somewhere else.
[[ "$PMT_DB" == /* ]] || PMT_DB="$PWD/$PMT_DB"

PMT_FETCH="${PMT_FETCH:-./fetch_corpus.sh}"
if [[ ! -f "$PMT_FETCH" ]]; then
  echo "ERROR: $PMT_FETCH not found. Submit from examples/sol_gaudi, where" >&2
  echo "       fetch_corpus.sh lives, or set PMT_FETCH to its path." >&2
  exit 1
fi

if [[ -f "$PMT_SECRETS" ]]; then
  echo "=== Sourcing credentials from $PMT_SECRETS ==="
  set +u
  source "$PMT_SECRETS"
  set -u
else
  echo "NOTE: no $PMT_SECRETS found; relying on whatever the job inherited."
fi

echo "=== Job ${SLURM_JOB_ID:-interactive} on $(hostname) ==="
echo "query=$PMT_QUERY db=$PMT_DB count=$PMT_COUNT"

# bash rather than direct execution, so a missing exec bit is not fatal.
exec bash "$PMT_FETCH" "$PMT_QUERY" "$PMT_DB" "$PMT_COUNT"
