// bench_ar.h
// AR decode benchmark: single-GPU and tensor-parallel.

#pragma once

#include "bench_common.h"
#include "config.h"

// Run the AR decode benchmark for a given model and TP degree.
// Generates N_tokens tokens with M=1 per step.
//
// Behavior is controlled by `opts`:
//   - opts.skip_component: skip the per-step (M=1) timing pass.
//   - opts.skip_e2e:       skip the N-token decode pass (component only).
//   - opts.csv:            also emit one CSV row at the end (prefixed RESULT,).
void run_ar_benchmark(
  const ModelConfig& cfg,
  int tp,
  int N_tokens,
  int warmup_iters,
  int bench_iters,
  const BenchOpts& opts
);
