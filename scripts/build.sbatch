#!/bin/bash
#SBATCH -J bench-build
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --gres=gpu:v100-32:1
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Build ./benchmark with NCCL on a compute node. Run once after a clean
# checkout (or after Makefile/source changes) before submitting any sweeps.

source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"

make clean
make USE_NCCL=1 \
     CXX_HOST="$(which g++)" \
     EXTRA_FLAGS="-I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib"

./benchmark --help | head -n 20
echo "Build OK."
