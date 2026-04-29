#!/bin/bash
#SBATCH -J roofline
#SBATCH -p GPU
#SBATCH -N 1
#SBATCH --gpus=v100-32:8
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Plot 6 (Section 3, Poster Session): roofline-style analysis. The
# component-timing block (`target_step_ms`, `target_verify_ms`,
# `draft_step_ms`, `draft_total_ms`, `prespec_ms`) gives us the per-stage
# kernel cost, which we plot against the V100 hardware roofline (peak
# fp16 flops, HBM2 BW) and against cuBLAS/NCCL microbenchmark numbers.
#
# Output: results/05_roofline.csv

source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"

OUT=results/05_roofline.csv
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
MODELS=(llama-1b llama-8b llama-70b)
TPS=(1 2 4 8)
KS=(1 2 4 8)
FS=(1 2 4 8)

for TARGET in "${MODELS[@]}"; do
  for TP in "${TPS[@]}"; do
    if [[ "$TARGET" == "llama-70b" && "$TP" -lt 8 ]]; then
      continue
    fi
    # AR: only target step.
    run --mode ar --target-model "$TARGET" --tp "$TP" \
        --N "$N" --warmup "$WARMUP" --iters "$ITERS"

    for K in "${KS[@]}"; do
      # SD: adds target_verify(M=K), draft_step, draft_total.
      run --mode sd --target-model "$TARGET" --draft-model llama-1b \
          --tp "$TP" --K "$K" --alpha "$ALPHA_FIXED" \
          --N "$N" --warmup "$WARMUP" --iters "$ITERS"

      # SSD: adds prespec(K, M=F).
      if (( TP <= 7 )); then
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
