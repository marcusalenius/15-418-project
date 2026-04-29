# Benchmark sbatch scripts (Bridges-2 V100-32)

These scripts produce the data for the plots listed in Section 3 (Poster
Session) of the milestone report. Each sweep writes a single CSV under
`results/`.

Following PSC's [benchmarking guidance](https://www.psc.edu/resources/bridges-2/user-guide/#benchmarking), every script grabs a
**whole `v100-32` node** (`-p GPU --gpus=v100-32:8`) so no other tenants
share the node and timings stay clean.

## One-time setup

From a Bridges-2 login node:

```bash
# Set up conda + NCCL (per run_on_psc.md, only needed once).
conda create -n myenv python=3.10 -y
conda activate myenv
conda install -c conda-forge nccl -y

cd $PROJECT/15418/project        # or wherever you cloned the repo
sbatch scripts/build.sbatch      # builds ./benchmark with USE_NCCL=1
```

If your conda install lives somewhere other than `~/miniconda3`, override
`CONDA_BASE` (e.g. `CONDA_BASE=$PROJECT/miniconda3 sbatch ...`).

## Running the sweeps

Submit individually:

```bash
sbatch scripts/01_sweep_alpha_K.sbatch
sbatch scripts/02_sweep_phit.sbatch
sbatch scripts/03_sweep_fanout.sbatch
sbatch scripts/04_breakdown.sbatch
sbatch scripts/05_roofline.sbatch
```

Or all at once (build first, then everything as `afterok` dependents):

```bash
bash scripts/submit_all.sh
```

Most knobs (model, TP, K, alpha grid, etc.) are environment variables at
the top of each script, so can override without editing:

```bash
TARGET=llama-70b TP=4 sbatch scripts/01_sweep_alpha_K.sbatch
```

## What maps to which plot


| Plot in the report                      | Script                            | Output CSV                 |
| --------------------------------------- | --------------------------------- | -------------------------- |
| 1. AR vs. SD vs. SSD over alpha and K   | `01_sweep_alpha_K.sbatch`         | `results/01_alpha_K.csv`   |
| 2. Cache hit rate sweep                 | `02_sweep_phit.sbatch`            | `results/02_phit.csv`      |
| 3. Fan-out sweep                        | `03_sweep_fanout.sbatch`          | `results/03_fanout.csv`    |
| 4. Per-round latency breakdown          | `04_breakdown.sbatch`             | `results/04_breakdown.csv` |
| 5. Verification window characterization | `04_breakdown.sbatch` (same data) | `results/04_breakdown.csv` |
| 6. Roofline-style analysis              | `05_roofline.sbatch`              | `results/05_roofline.csv`  |
| 7. Batch size scaling                   | *not yet supported*               | —                          |
| 8. Throughput-latency Pareto frontier   | *not yet supported*               | —                          |


