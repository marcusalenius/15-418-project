#!/bin/bash
# Submit all sweep scripts in dependency order. The build job runs first,
# and every sweep waits for it to succeed (-d afterok:<jobid>).
#
# Usage (from project root):
#   bash scripts/submit_all.sh
#
# Drop the build step if already built the binary:
#   SKIP_BUILD=1 bash scripts/submit_all.sh

set -euo pipefail
cd "$(dirname "$0")/.."

SKIP_BUILD=${SKIP_BUILD:-0}
DEP=""

if [[ "$SKIP_BUILD" != "1" ]]; then
  BUILD_ID=$(sbatch --parsable scripts/build.sbatch)
  echo "Submitted build: $BUILD_ID"
  DEP="-d afterok:$BUILD_ID"
fi

for s in \
  scripts/01_sweep_alpha_K.sbatch \
  scripts/02_sweep_phit.sbatch \
  scripts/03_sweep_fanout.sbatch \
  scripts/04_breakdown.sbatch \
  scripts/05_roofline.sbatch
do
  jid=$(sbatch --parsable $DEP "$s")
  echo "Submitted $s -> $jid"
done

echo "Run 'squeue -u \$USER' to monitor."
