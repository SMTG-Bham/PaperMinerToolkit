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
#   sbatch --export=ALL,PM_COUNT=500 fetch_corpus.sbatch   # bound it
#   sbatch --export=ALL,PM_QUERY="polymer biodegradation OECD 310",PM_DB=bio.db \
#          fetch_corpus.sbatch
#
# The log (ps-fetch-<jobid>.log) and the corpus land beside the scripts, where
# scrape_gaudi.sbatch looks for them.
#
# Credentials: a batch job does not reliably inherit your shell environment, so
# keep the keys in a file only you can read and this script will source it:
#
#   printf 'export ELSEVIER_API_KEY=...\nexport CORE_API_KEY=...\nexport UNPAYWALL_EMAIL=you@asu.edu\nexport OPENALEX_API_KEY=...\n' > ~/.paperminer_env
#   chmod 600 ~/.paperminer_env
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
# Mind that against the unbounded PM_COUNT below. Search is cheap and re-running
# it is nearly free, because pm_search upserts on DOI. Download is not: pm_download
# walks every row in the corpus and has no skip for assets it already has, so a
# job killed at the wall clock re-downloads from the start on the next submit.
# If the corpus is big enough for that to bite, either bound PM_COUNT so the job
# finishes inside four hours, or move to a partition that allows longer:
#
#   #SBATCH -p general
#   #SBATCH -t 1-00:00:00

set -euo pipefail

PM_QUERY="${PM_QUERY:-biodegradable polymer OECD 301 biodegradation}"
PM_DB="${PM_DB:-papers.db}"
# Every paper the sources will give us, per source. See fetch_corpus.sh for what
# the number means; set it to something small while testing a new query.
PM_COUNT="${PM_COUNT:-1000000}"
PM_SECRETS="${PM_SECRETS:-$HOME/.paperminer_env}"

# Run from wherever the job was submitted — normally examples/sol_gaudi — so the
# corpus lands beside the scripts rather than in the scheduler's spool directory.
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# fetch_corpus.sh anchors a relative database to its own directory. Resolve
# PM_DB here so this job's working directory wins even if PM_FETCH points at a
# copy of the script kept somewhere else.
[[ "$PM_DB" == /* ]] || PM_DB="$PWD/$PM_DB"

PM_FETCH="${PM_FETCH:-./fetch_corpus.sh}"
if [[ ! -f "$PM_FETCH" ]]; then
  echo "ERROR: $PM_FETCH not found. Submit from examples/sol_gaudi, where" >&2
  echo "       fetch_corpus.sh lives, or set PM_FETCH to its path." >&2
  exit 1
fi

if [[ -f "$PM_SECRETS" ]]; then
  echo "=== Sourcing credentials from $PM_SECRETS ==="
  set +u
  source "$PM_SECRETS"
  set -u
else
  echo "NOTE: no $PM_SECRETS found; relying on whatever the job inherited."
fi

echo "=== Job ${SLURM_JOB_ID:-interactive} on $(hostname) ==="
echo "query=$PM_QUERY db=$PM_DB count=$PM_COUNT"

# bash rather than direct execution, so a missing exec bit is not fatal.
exec bash "$PM_FETCH" "$PM_QUERY" "$PM_DB" "$PM_COUNT"
