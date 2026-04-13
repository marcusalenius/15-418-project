// forward_tp.cu
// Multi-GPU tensor-parallel forward pass implementations.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <nccl.h>

#include "check.h"
#include "gemm.h"
#include "forward_tp.h"

// ---------------------------------------------------------------------------
// Weight allocation
// ---------------------------------------------------------------------------
size_t alloc_tp_layer_weights(
  TPLayerWeights& w,
  const ModelConfig& cfg,
  int T
) {
  size_t total = 0;

  auto alloc = [&](half** ptr, int rows, int cols) {
    size_t bytes = static_cast<size_t>(rows) * cols * sizeof(half);
    CHECK_CUDA(cudaMalloc(ptr, bytes));
    CHECK_CUDA(cudaMemset(*ptr, 0x3C, bytes));
    total += bytes;
  };

  int d    = cfg.d_model;
  int d_q  = cfg.q_dim()  / T;
  int d_kv = cfg.kv_dim() / T;
  int d_ff = cfg.d_ff    / T;

  alloc(&w.W_q,    d, d_q);
  alloc(&w.W_k,    d, d_kv);
  alloc(&w.W_v,    d, d_kv);
  alloc(&w.W_o,    d_q, d);
  alloc(&w.W_gate, d, d_ff);
  alloc(&w.W_up,   d, d_ff);
  alloc(&w.W_down, d_ff, d);

  return total;
}

void free_tp_layer_weights(TPLayerWeights& w) {
  cudaFree(w.W_q);
  cudaFree(w.W_k);
  cudaFree(w.W_v);
  cudaFree(w.W_o);
  cudaFree(w.W_gate);
  cudaFree(w.W_up);
  cudaFree(w.W_down);
}

// ---------------------------------------------------------------------------
// Forward passes
// ---------------------------------------------------------------------------

void forward_layer_tp(
  cublasHandle_t handle,
  ncclComm_t comm,
  cudaStream_t stream,
  const ModelConfig& cfg,
  const TPLayerWeights& w,
  int T,
  int M,
  half* x,
  half* scratch1,
  half* scratch2
) {
  int d    = cfg.d_model;
  int d_q  = cfg.q_dim()  / T;
  int d_kv = cfg.kv_dim() / T;
  int d_ff = cfg.d_ff    / T;

  // --- Attention projections (column-parallel) ---
  hgemm(handle, M, d_q,  d, x, w.W_q, scratch1);
  hgemm(handle, M, d_kv, d, x, w.W_k, scratch2);
  hgemm(handle, M, d_kv, d, x, w.W_v, scratch2);

  // O (row-parallel, partial sum)
  hgemm(handle, M, d, d_q, scratch1, w.W_o, x);

  // All-reduce after attention
  CHECK_NCCL(ncclAllReduce(
    x, x, static_cast<size_t>(M) * d,
    ncclFloat16, ncclSum, comm, stream
  ));

  // --- FFN projections (column-parallel) ---
  hgemm(handle, M, d_ff, d, x, w.W_gate, scratch1);
  hgemm(handle, M, d_ff, d, x, w.W_up,   scratch2);

  // Down (row-parallel, partial sum)
  hgemm(handle, M, d, d_ff, scratch1, w.W_down, x);

  // All-reduce after FFN
  CHECK_NCCL(ncclAllReduce(
    x, x, static_cast<size_t>(M) * d,
    ncclFloat16, ncclSum, comm, stream
  ));
}

void forward_model_tp(
  cublasHandle_t handle,
  ncclComm_t comm,
  cudaStream_t stream,
  const ModelConfig& cfg,
  TPLayerWeights* all_weights,
  int T,
  int M,
  half* x,
  half* scratch1,
  half* scratch2
) {
  for (int l = 0; l < cfg.n_layers; l++)
    forward_layer_tp(
      handle, comm, stream, cfg, all_weights[l], T, M, x, scratch1, scratch2
    );
}

// ---------------------------------------------------------------------------
// Activation buffers (TP-aware)
// ---------------------------------------------------------------------------
ActivationBuffers alloc_activations_tp(const ModelConfig& cfg, int M, int T) {
  ActivationBuffers buf;
  int max_local = std::max(cfg.q_dim() / T, cfg.d_ff / T);

  CHECK_CUDA(cudaMalloc(
    &buf.x, static_cast<size_t>(M) * cfg.d_model * sizeof(half)
  ));
  CHECK_CUDA(cudaMalloc(
    &buf.scratch1, static_cast<size_t>(M) * max_local * sizeof(half)
  ));
  CHECK_CUDA(cudaMalloc(
    &buf.scratch2, static_cast<size_t>(M) * max_local * sizeof(half)
  ));
  CHECK_CUDA(cudaMemset(
    buf.x, 0x3C, static_cast<size_t>(M) * cfg.d_model * sizeof(half)
  ));
  return buf;
}
