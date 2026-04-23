// bench_ssd.h
// SSD benchmark: tensor-parallel target + dedicated draft GPU.

#pragma once

#include "config.h"

void run_ssd_benchmark(
  const ModelConfig& target_model,
  const ModelConfig& draft_model,
  int tp,
  int K,
  int F,
  int N,
  float alpha,
  float phit,
  int warmup,
  int iters
);
