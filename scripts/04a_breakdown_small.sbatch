#!/bin/bash
#SBATCH -J breakdown-small
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --gpus=v100-32:4
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Plot 4 (per-round latency breakdown) and Plot 5 (verification window
# characterization). Reduced-config version for 4-GPU GPU-shared interactive
# sessions when the full GPU partition is not available before the deadline.
#
# Restrictions vs the original 04_breakdown.sbatch:
#   - Drops the llama-70b/tp=8 config (needs 8 GPUs, full node only).
#   - SSD on 4 GPUs only works for tp+1 <= 4, i.e., tp in {1, 2} since tp=3
#     is not divisible across the model dimensions. AR/SD still run at tp=4.
#
# Output: results/04a_breakdown_small.csv

source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"

OUT=results/04a_breakdown_small.csv
N=${N:-128}
WARMUP=${WARMUP:-10}
ITERS=${ITERS:-30}
ALPHA_FIXED=${ALPHA_FIXED:-0.90}
K_FIXED=${K_FIXED:-4}
F_FIXED=${F_FIXED:-4}
PHIT_FIXED=${PHIT_FIXED:-0.85}

./benchmark --csv-header > "$OUT"

run() {
  echo "[run] $*"
  ./benchmark "$@" --csv | grep '^RESULT,' >> "$OUT"
}

# Config matrix: (target, draft, tp).
# Note: SSD requires tp+1 GPUs. With 4 GPUs available, tp_max=4 for AR/SD
# and tp_max=3 for SSD. Since tp=3 isn't a valid TP for these models, SSD
# is effectively limited to tp in {1, 2}.
CONFIGS=(
  "llama-1b llama-1b 1"
  "llama-8b llama-1b 1"
  "llama-8b llama-1b 2"
  "llama-8b llama-1b 4"
)

for cfg in "${CONFIGS[@]}"; do
  read -r TARGET DRAFT TP <<<"$cfg"

  run --mode ar  --target-model "$TARGET" --tp "$TP" \
      --N "$N" --warmup "$WARMUP" --iters "$ITERS"

  run --mode sd  --target-model "$TARGET" --draft-model "$DRAFT" \
      --tp "$TP" --K "$K_FIXED" --alpha "$ALPHA_FIXED" \
      --N "$N" --warmup "$WARMUP" --iters "$ITERS"

  # SSD on 4 GPUs requires tp+1 <= 4.
  if (( TP + 1 <= 4 )); then
    run --mode ssd --target-model "$TARGET" --draft-model "$DRAFT" \
        --tp "$TP" --K "$K_FIXED" --F "$F_FIXED" \
        --alpha "$ALPHA_FIXED" --phit "$PHIT_FIXED" \
        --N "$N" --warmup "$WARMUP" --iters "$ITERS"
  fi
done

echo "Done. Wrote $(wc -l < "$OUT") lines to $OUT."
