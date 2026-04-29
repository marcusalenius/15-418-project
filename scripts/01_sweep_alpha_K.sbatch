#!/bin/bash
#SBATCH -J alpha-K-sweep
#SBATCH -p GPU
#SBATCH -N 1
#SBATCH --gpus=v100-32:8
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Plot 1 (Section 3, Poster Session): AR vs SD vs SSD throughput across alpha
# and K, all three side-by-side. AR is independent of alpha/K so it runs once.
# SSD has phit and F fixed; vary those in the dedicated sweeps below.
#
# Output: results/01_alpha_K.csv (one RESULT row per config).

source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"

OUT=results/01_alpha_K.csv
TARGET=${TARGET:-llama-8b}
DRAFT=${DRAFT:-llama-1b}
TP=${TP:-1}
N=${N:-128}
WARMUP=${WARMUP:-10}
ITERS=${ITERS:-30}
PHIT_FIXED=${PHIT_FIXED:-0.85}
F_FIXED=${F_FIXED:-4}

ALPHAS=(0.30 0.50 0.70 0.80 0.90 0.95 1.00)
KS=(1 2 4 6 8)

./benchmark --csv-header > "$OUT"

run() {
  echo "[run] $*"
  ./benchmark "$@" --csv | grep '^RESULT,' >> "$OUT"
}

# AR baseline (no alpha/K dependence).
run --mode ar  --target-model "$TARGET" --tp "$TP" \
    --N "$N" --warmup "$WARMUP" --iters "$ITERS"

# SD and SSD across the alpha x K grid.
for A in "${ALPHAS[@]}"; do
  for K in "${KS[@]}"; do
    run --mode sd  --target-model "$TARGET" --draft-model "$DRAFT" \
        --tp "$TP" --K "$K" --alpha "$A" \
        --N "$N" --warmup "$WARMUP" --iters "$ITERS"

    run --mode ssd --target-model "$TARGET" --draft-model "$DRAFT" \
        --tp "$TP" --K "$K" --F "$F_FIXED" --alpha "$A" --phit "$PHIT_FIXED" \
        --N "$N" --warmup "$WARMUP" --iters "$ITERS"
  done
done

echo "Done. Wrote $(wc -l < "$OUT") lines to $OUT."
