#!/bin/bash
# Common environment setup for all PSC sbatch scripts.
# Source this from inside an sbatch script. Assumes:
#   - You ran `sbatch scripts/foo.sbatch` from the project root, so
#     $SLURM_SUBMIT_DIR is the project root.
#   - The NCCL/cuBLAS-enabled build steps from run_on_psc.md have been done
#     once already (i.e. `myenv` conda env exists and `./benchmark` is built
#     with USE_NCCL=1). If not, run scripts/build.sbatch first.

set -euo pipefail

module purge
module load cuda

# Conda activation (per run_on_psc.md). Override CONDA_BASE / CONDA_ENV via
# the environment if your install or env name differ. We try, in order:
#   1. $CONDA_BASE (explicit override)
#   2. $HOME/miniconda3, $HOME/anaconda3
#   3. The PSC Bridges-2 system anaconda
#   4. Whatever `conda` is on PATH (resolved to its install root)
CONDA_ENV="${CONDA_ENV:-myenv}"
if [[ -z "${CONDA_BASE:-}" ]]; then
  for cand in \
    "$HOME/miniconda3" \
    "$HOME/anaconda3" \
    "/opt/packages/anaconda3-2024.10-1"; do
    if [[ -f "$cand/etc/profile.d/conda.sh" ]]; then
      CONDA_BASE="$cand"
      break
    fi
  done
fi
if [[ -z "${CONDA_BASE:-}" ]] && command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(dirname "$(dirname "$(command -v conda)")")"
fi
if [[ -z "${CONDA_BASE:-}" || ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  echo "ERROR: could not locate a conda installation" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

mkdir -p results logs

echo "===== job: ${SLURM_JOB_NAME:-?} (${SLURM_JOB_ID:-?}) ====="
echo "host:   $(hostname)"
echo "cwd:    $(pwd)"
echo "date:   $(date)"
nvidia-smi || true
echo "================================================="
