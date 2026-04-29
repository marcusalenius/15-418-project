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

# Conda activation (per run_on_psc.md). Adjust CONDA_BASE if your install
# is elsewhere (e.g. $PROJECT/miniconda3).
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-myenv}"
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

mkdir -p results logs

echo "===== job: ${SLURM_JOB_NAME:-?} (${SLURM_JOB_ID:-?}) ====="
echo "host:   $(hostname)"
echo "cwd:    $(pwd)"
echo "date:   $(date)"
nvidia-smi || true
echo "================================================="
