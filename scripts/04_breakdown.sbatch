#!/bin/bash
#SBATCH -J breakdown
#SBATCH -p GPU
#SBATCH -N 1
#SBATCH --gpus=v100-32:8
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Plot 4 (per-round latency breakdown) and Plot 5 (verification window
# characterization, i.e. how many draft passes fit inside one target verify).
# Both come from the per-stage component timings. We run AR, SD, SSD at a 
# representative config for several model sizes / TP combinations so the 
# stacked-bar chart has the configurations we care about.
#
# Output: results/04_breakdown.csv

source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"

OUT=results/04_breakdown.csv
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
# Note: SSD requires tp+1 GPUs. With one v100-32 node (8 GPUs), tp_max=7
# for SSD and tp_max=8 for AR/SD.
CONFIGS=(
  "llama-1b llama-1b 1"
  "llama-8b llama-1b 1"
  "llama-8b llama-1b 2"
  "llama-8b llama-1b 4"
  "llama-70b llama-1b 8"
)

for cfg in "${CONFIGS[@]}"; do
  read -r TARGET DRAFT TP <<<"$cfg"

  run --mode ar  --target-model "$TARGET" --tp "$TP" \
      --N "$N" --warmup "$WARMUP" --iters "$ITERS"

  run --mode sd  --target-model "$TARGET" --draft-model "$DRAFT" \
      --tp "$TP" --K "$K_FIXED" --alpha "$ALPHA_FIXED" \
      --N "$N" --warmup "$WARMUP" --iters "$ITERS"

  # SSD with tp greater than 7 will exceed the 8 GPUs on a v100-32 node.
  if (( TP <= 7 )); then
    run --mode ssd --target-model "$TARGET" --draft-model "$DRAFT" \
        --tp "$TP" --K "$K_FIXED" --F "$F_FIXED" \
        --alpha "$ALPHA_FIXED" --phit "$PHIT_FIXED" \
        --N "$N" --warmup "$WARMUP" --iters "$ITERS"
  fi
done

echo "Done. Wrote $(wc -l < "$OUT") lines to $OUT."
