// bench_ar.h
// AR decode benchmark: single-GPU and tensor-parallel.

#pragma once

#include "config.h"

// Run the full AR decode benchmark for a given model and TP degree.
// Prints AR decode throughput and verification pass table.
void run_ar_benchmark(
  const ModelConfig& cfg,
  int tp,
  int N_tokens,
  int warmup_iters,
  int bench_iters
);
