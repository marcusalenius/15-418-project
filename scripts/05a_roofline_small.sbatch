#!/bin/bash
#SBATCH -J roofline-small
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --gpus=v100-32:4
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Plot 6 (Section 3, Poster Session): roofline-style analysis. Reduced-config
# version for 4-GPU GPU-shared sessions when the full GPU partition is not
# available before the deadline.
#
# Restrictions vs the original 05_roofline.sbatch:
#   - Drops llama-70b entirely (needs tp=8, full node only).
#   - TP set restricted to {1, 2, 4}. tp=3 isn't divisible across these
#     models. tp=8 needs the full node.
#   - SSD on 4 GPUs requires tp+1 <= 4, so SSD is effectively limited to
#     tp in {1, 2}.
#
# Output: results/05a_roofline_small.csv

source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"

OUT=results/05a_roofline_small.csv
N=${N:-128}
WARMUP=${WARMUP:-15}
ITERS=${ITERS:-30}
ALPHA_FIXED=${ALPHA_FIXED:-0.90}
PHIT_FIXED=${PHIT_FIXED:-0.85}

./benchmark --csv-header > "$OUT"

run() {
  echo "[run] $*"
  ./benchmark "$@" --skip-e2e --csv | grep '^RESULT,' >> "$OUT"
}

# Sweep model size x TP x K (and F for SSD)
MODELS=(llama-1b llama-8b)
TPS=(1 2 4)
KS=(1 4 8)
FS=(1 4 8)

for TARGET in "${MODELS[@]}"; do
  for TP in "${TPS[@]}"; do
    # AR: only target step.
    run --mode ar --target-model "$TARGET" --tp "$TP" \
        --N "$N" --warmup "$WARMUP" --iters "$ITERS"

    for K in "${KS[@]}"; do
      # SD: adds target_verify(M=K), draft_step, draft_total.
      run --mode sd --target-model "$TARGET" --draft-model llama-1b \
          --tp "$TP" --K "$K" --alpha "$ALPHA_FIXED" \
          --N "$N" --warmup "$WARMUP" --iters "$ITERS"

      # SSD on 4 GPUs requires tp+1 <= 4.
      if (( TP + 1 <= 4 )); then
        for F in "${FS[@]}"; do
          run --mode ssd --target-model "$TARGET" --draft-model llama-1b \
              --tp "$TP" --K "$K" --F "$F" \
              --alpha "$ALPHA_FIXED" --phit "$PHIT_FIXED" \
              --N "$N" --warmup "$WARMUP" --iters "$ITERS"
        done
      fi
    done
  done
done

echo "Done. Wrote $(wc -l < "$OUT") lines to $OUT."
