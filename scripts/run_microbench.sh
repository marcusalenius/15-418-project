#!/bin/bash
# scripts/run_microbench.sh
# Run NCCL (all-reduce, send/recv) and cuBLAS (GEMM sweep) microbenchmarks
# from inside the 15-418-project root on a PSC interactive session.
#
# Outputs (to ./microbench/):
#   nccl_allreduce.csv  - all_reduce_perf, np in {2,3,4} x size sweep
#   nccl_sendrecv.csv   - sendrecv_perf, np=2, size sweep
#   cublas_gemm.csv     - gemm_micro, single-GPU GEMM sweep
#   raw/                - raw stdout from each tool
#
# Usage:
#   bash scripts/run_microbench.sh

set -euo pipefail

OUT_DIR="${OUT_DIR:-microbench}"
RAW_DIR="$OUT_DIR/raw"
NCCL_TESTS_DIR="${NCCL_TESTS_DIR:-../nccl-tests}"
GEMM_SRC="${GEMM_SRC:-scripts/gemm_micro.cu}"
GEMM_BIN="${GEMM_BIN:-build/gemm_micro}"

MIN_BYTES="${MIN_BYTES:-8}"
MAX_BYTES="${MAX_BYTES:-128M}"
STEP="${STEP:-2}"

mkdir -p "$OUT_DIR" "$RAW_DIR"

# ---------------------------------------------------------------------------
# Environment (same as scripts/_env.sh: cuda + myenv conda for NCCL)
# ---------------------------------------------------------------------------
if [[ -f scripts/_env.sh ]]; then
  # shellcheck disable=SC1091
  source scripts/_env.sh
else
  module purge
  module load cuda
fi

NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
echo "[env] $NUM_GPUS GPUs visible, NCCL_TESTS_DIR=$NCCL_TESTS_DIR"
(( NUM_GPUS < 2 )) && { echo "need >=2 GPUs"; exit 1; }

# ---------------------------------------------------------------------------
# Build nccl-tests WITHOUT MPI (single-node, threads via -g)
# Force a clean rebuild if the existing build was attempted with MPI=1.
# ---------------------------------------------------------------------------
NCCL_HOME_LOCAL="${NCCL_HOME:-${CONDA_PREFIX:-}}"
if [[ ! -x "$NCCL_TESTS_DIR/build/all_reduce_perf" ]]; then
  echo "[build] nccl-tests (no MPI)"
  ( cd "$NCCL_TESTS_DIR" && \
    make clean >/dev/null 2>&1 || true && \
    make -j CUDA_HOME="$CUDA_HOME" NCCL_HOME="$NCCL_HOME_LOCAL" )
fi

# ---------------------------------------------------------------------------
# Build gemm_micro
# ---------------------------------------------------------------------------
if [[ ! -x "$GEMM_BIN" && -f "$GEMM_SRC" ]]; then
  echo "[build] $GEMM_SRC -> $GEMM_BIN"
  mkdir -p "$(dirname "$GEMM_BIN")"
  nvcc -O2 -std=c++17 -o "$GEMM_BIN" "$GEMM_SRC" -lcublas
fi

# ---------------------------------------------------------------------------
# Parser: nccl-tests data rows -> CSV
# Cols: size count type redop root  time(us) algbw busbw #wrong  in-place(time algbw busbw #wrong)
# We take out-of-place: size=$1, time=$6, algbw=$7, busbw=$8.
# ---------------------------------------------------------------------------
parse_nccl_log_to_csv() {
  local logfile="$1" op="$2" np="$3"
  awk -v op="$op" -v np="$np" '
    /^[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]+(float|half|int8|uint8|int|uint32|int32|int64|uint64|double)/ {
      print op "," np "," $1 "," $6 "," $7 "," $8
    }' "$logfile"
}

# ---------------------------------------------------------------------------
# NCCL all-reduce: -g NP spawns NP threads in one process (no MPI needed)
# ---------------------------------------------------------------------------
echo "op,np,size_bytes,time_us,algbw_GBs,busbw_GBs" > "$OUT_DIR/nccl_allreduce.csv"
for NP in 2 3 4; do
  (( NP > NUM_GPUS )) && continue
  echo "[nccl] all_reduce np=$NP"
  RAW="$RAW_DIR/allreduce_np${NP}.log"
  "$NCCL_TESTS_DIR/build/all_reduce_perf" \
    -b "$MIN_BYTES" -e "$MAX_BYTES" -f "$STEP" -g "$NP" \
    > "$RAW" 2>&1 || { echo "  failed, see $RAW"; continue; }
  parse_nccl_log_to_csv "$RAW" "all_reduce" "$NP" >> "$OUT_DIR/nccl_allreduce.csv"
done

# ---------------------------------------------------------------------------
# NCCL send/recv: -g 2
# ---------------------------------------------------------------------------
echo "op,np,size_bytes,time_us,algbw_GBs,busbw_GBs" > "$OUT_DIR/nccl_sendrecv.csv"
echo "[nccl] sendrecv np=2"
RAW="$RAW_DIR/sendrecv_np2.log"
"$NCCL_TESTS_DIR/build/sendrecv_perf" \
  -b "$MIN_BYTES" -e "$MAX_BYTES" -f "$STEP" -g 2 \
  > "$RAW" 2>&1 || echo "  failed, see $RAW"
parse_nccl_log_to_csv "$RAW" "sendrecv" "2" >> "$OUT_DIR/nccl_sendrecv.csv"

# ---------------------------------------------------------------------------
# cuBLAS GEMM sweep
# ---------------------------------------------------------------------------
if [[ -x "$GEMM_BIN" ]]; then
  echo "[cublas] gemm sweep on GPU 0"
  RAW="$RAW_DIR/gemm_micro_gpu0.log"
  CUDA_VISIBLE_DEVICES=0 "$GEMM_BIN" --csv > "$RAW" 2>&1 \
    || echo "  gemm_micro had errors, see $RAW"
  cp "$RAW" "$OUT_DIR/cublas_gemm.csv"
  echo "[cublas] $(($(wc -l < "$OUT_DIR/cublas_gemm.csv") - 1)) rows"
else
  echo "[cublas] $GEMM_BIN not found — drop $GEMM_SRC and rerun"
fi

echo
echo "[done] outputs:"
ls -la "$OUT_DIR"/*.csv 2>/dev/null || true
echo "raw logs in $RAW_DIR/"
