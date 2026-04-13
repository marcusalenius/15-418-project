// gpu_context.h
// Shared GPU utilities for benchmark modules.

#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "check.h"
#include "config.h"

// ---------------------------------------------------------------------------
// GPU memory
// ---------------------------------------------------------------------------

inline size_t get_free_gpu_memory(int device = 0) {
  CHECK_CUDA(cudaSetDevice(device));
  size_t free_mem, total_mem;
  CHECK_CUDA(cudaMemGetInfo(&free_mem, &total_mem));
  return free_mem;
}

inline bool model_fits(const ModelConfig& cfg, int tp) {
  size_t free_mem = get_free_gpu_memory(0);
  size_t weight_bytes = (tp == 1)
    ? cfg.total_weight_bytes()
    : cfg.total_tp_weight_bytes(tp);
  return weight_bytes < static_cast<size_t>(free_mem * 0.85);
}

// ---------------------------------------------------------------------------
// Per-GPU context for TP benchmarks
// ---------------------------------------------------------------------------

#ifdef USE_NCCL
#include <nccl.h>

struct GPUContext {
  cublasHandle_t handle;
  ncclComm_t comm;
  cudaStream_t stream;
};

inline GPUContext init_gpu(int rank, int T, ncclUniqueId nccl_id) {
  GPUContext ctx;
  CHECK_CUDA(cudaSetDevice(rank));
  CHECK_NCCL(ncclCommInitRank(&ctx.comm, T, nccl_id, rank));
  CHECK_CUBLAS(cublasCreate(&ctx.handle));
  CHECK_CUBLAS(cublasSetMathMode(ctx.handle, CUBLAS_TENSOR_OP_MATH));
  CHECK_CUDA(cudaStreamCreate(&ctx.stream));
  CHECK_CUBLAS(cublasSetStream(ctx.handle, ctx.stream));
  return ctx;
}

inline void destroy_gpu(GPUContext& ctx) {
  CHECK_CUBLAS(cublasDestroy(ctx.handle));
  CHECK_CUDA(cudaStreamDestroy(ctx.stream));
  ncclCommDestroy(ctx.comm);
}

#endif // USE_NCCL