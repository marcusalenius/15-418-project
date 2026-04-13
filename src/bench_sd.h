// bench_sd.h
// SD benchmark: single-GPU and tensor-parallel.

#pragma once

#include "config.h"

void run_sd_benchmark(
  const ModelConfig& target_model, 
  const ModelConfig& draft_model, 
  int tp, 
  int K, 
  int N, 
  float alpha,
  int warmup, 
  int iters
);