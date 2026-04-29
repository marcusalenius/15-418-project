#!/bin/bash
#SBATCH -J phit-sweep
#SBATCH -p GPU
#SBATCH -N 1
#SBATCH --gpus=v100-32:8
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Plot 2 (Section 3, Poster Session): cache-hit-rate sweep. Show how SSD's
# advantage over SD changes as p_hit varies, with everything else fixed.
# We include AR and one SD config as baselines for context.
#
# Output: results/02_phit.csv

source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"

OUT=results/02_phit.csv
TARGET=${TARGET:-llama-8b}
DRAFT=${DRAFT:-llama-1b}
TP=${TP:-1}
N=${N:-128}
WARMUP=${WARMUP:-10}
ITERS=${ITERS:-30}
ALPHA_FIXED=${ALPHA_FIXED:-0.90}
K_FIXED=${K_FIXED:-4}
F_FIXED=${F_FIXED:-4}

PHITS=(0.30 0.50 0.70 0.80 0.85 0.90 0.95 1.00)

./benchmark --csv-header > "$OUT"

run() {
  echo "[run] $*"
  ./benchmark "$@" --csv | grep '^RESULT,' >> "$OUT"
}

run --mode ar  --target-model "$TARGET" --tp "$TP" \
    --N "$N" --warmup "$WARMUP" --iters "$ITERS"

run --mode sd  --target-model "$TARGET" --draft-model "$DRAFT" \
    --tp "$TP" --K "$K_FIXED" --alpha "$ALPHA_FIXED" \
    --N "$N" --warmup "$WARMUP" --iters "$ITERS"

for PHIT in "${PHITS[@]}"; do
  run --mode ssd --target-model "$TARGET" --draft-model "$DRAFT" \
      --tp "$TP" --K "$K_FIXED" --F "$F_FIXED" \
      --alpha "$ALPHA_FIXED" --phit "$PHIT" \
      --N "$N" --warmup "$WARMUP" --iters "$ITERS"
done

echo "Done. Wrote $(wc -l < "$OUT") lines to $OUT."
