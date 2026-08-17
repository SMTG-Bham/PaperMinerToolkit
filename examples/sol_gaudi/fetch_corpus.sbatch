#!/bin/bash
# Submit the biodegradable polymer corpus build as a batch job.
#
# This is a wrapper around fetch_corpus.sh for when the corpus is too large to
# babysit in an interactive session. Search and download are rate-limit bound
# rather than compute bound, so this asks for few cores and a lot of wall clock.
#
#   sbatch examples/sol_gaudi/fetch_corpus.sbatch
#   sbatch --export=ALL,PS_COUNT=500 examples/sol_gaudi/fetch_corpus.sbatch
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
# htc caps wall time at 4 hours but runs without interruption. For a corpus large
# enough to need longer, switch to the general partition, which allows up to a
# week on the public QOS:
#
#   #SBATCH -p general
#   #SBATCH -t 1-00:00:00

set -euo pipefail

PS_QUERY="${PS_QUERY:-biodegradable polymer OECD 301 biodegradation}"
PS_DB="${PS_DB:-papers.db}"
PS_COUNT="${PS_COUNT:-50}"
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
