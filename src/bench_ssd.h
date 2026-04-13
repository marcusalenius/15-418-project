// bench_ssd.h
// SSD benchmark: single-GPU and tensor-parallel.

#pragma once

#include "config.h"

void run_ssd_benchmark(
  const ModelConfig& target_model, 
  const ModelConfig& draft_model, 
  int tp, 
  int K, 
  int N, 
  int warmup, 
  int iters
);