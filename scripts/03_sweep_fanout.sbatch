#!/bin/bash
#SBATCH -J fanout-sweep
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --gpus=v100-32:2
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Plot 3 (Section 3, Poster Session): fan-out sweep. Captures the trade-off
# between pre-speculation work (~ K * draft forward at M=F) and the cache hit
# rate it produces. We sweep F at a fixed alpha/K and a few representative
# phit values so the trade-off curve has multiple lines.
#
# Output: results/03_fanout.csv

source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"

OUT=results/03_fanout.csv
TARGET=${TARGET:-llama-8b}
DRAFT=${DRAFT:-llama-1b}
TP=${TP:-1}
N=${N:-128}
WARMUP=${WARMUP:-10}
ITERS=${ITERS:-30}
ALPHA_FIXED=${ALPHA_FIXED:-0.90}
K_FIXED=${K_FIXED:-4}

FS=(1 2 3 4 6 8 12)
PHITS=(0.70 0.85 0.95)

./benchmark --csv-header > "$OUT"

run() {
  echo "[run] $*"
  ./benchmark "$@" --csv | grep '^RESULT,' >> "$OUT"
}

for PHIT in "${PHITS[@]}"; do
  for F in "${FS[@]}"; do
    run --mode ssd --target-model "$TARGET" --draft-model "$DRAFT" \
        --tp "$TP" --K "$K_FIXED" --F "$F" \
        --alpha "$ALPHA_FIXED" --phit "$PHIT" \
        --N "$N" --warmup "$WARMUP" --iters "$ITERS"
  done
done

echo "Done. Wrote $(wc -l < "$OUT") lines to $OUT."
