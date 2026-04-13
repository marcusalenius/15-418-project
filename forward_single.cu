// forward_single.cu
// Single-GPU forward pass and benchmarking implementations.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include "check.h"
#include "gemm.h"
#include "forward_single.h"

// ---------------------------------------------------------------------------
// Weight allocation
// ---------------------------------------------------------------------------

// Allocate weight matrices for one layer, fill with ~1.0
size_t alloc_layer_weights(LayerWeights& w, const ModelConfig& cfg) {
  size_t total = 0;

  auto alloc = [&](half** ptr, int rows, int cols) {
    size_t bytes = static_cast<size_t>(rows) * cols * sizeof(half);
    CHECK_CUDA(cudaMalloc(ptr, bytes));
    CHECK_CUDA(cudaMemset(*ptr, 0x3C, bytes)); // ~1.0 in FP16
    total += bytes;
  };

  int d    = cfg.d_model;
  int d_q  = cfg.q_dim();
  int d_kv = cfg.kv_dim();
  int d_ff = cfg.d_ff;

  alloc(&w.W_q,    d, d_q);
  alloc(&w.W_k,    d, d_kv);
  alloc(&w.W_v,    d, d_kv);
  alloc(&w.W_o,    d_q, d);
  alloc(&w.W_gate, d, d_ff);
  alloc(&w.W_up,   d, d_ff);
  alloc(&w.W_down, d_ff, d);

  return total;
}

void free_layer_weights(LayerWeights& w) {
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
void forward_layer(
  cublasHandle_t handle,
  const ModelConfig& cfg,
  const LayerWeights& w,
  int M,
  half* x,
  half* scratch1,
  half* scratch2
) {
  int d    = cfg.d_model;
  int d_q  = cfg.q_dim();
  int d_kv = cfg.kv_dim();
  int d_ff = cfg.d_ff;

  // --- Attention projections ---
  // Q: [M, d] x [d, d_q] (writing to scratch1)
  hgemm(handle, M, d_q,  d, x, w.W_q, scratch1);
  // K: [M, d] x [d, d_kv] (writing to scratch2)
  hgemm(handle, M, d_kv, d, x, w.W_k, scratch2);
  // V: [M, d] x [d, d_kv] (writing to scratch2)
  hgemm(handle, M, d_kv, d, x, w.W_v, scratch2);
  // O: [M, d_q] x [d_q, d] (reading from scratch1, writing to x)
  hgemm(handle, M, d, d_q, scratch1, w.W_o, x);

  // --- FFN projections ---
  // Gate: [M, d] x [d, d_ff] (writing to scratch1)
  hgemm(handle, M, d_ff, d, x, w.W_gate, scratch1);
  // Up: [M, d] x [d, d_ff] (writing to scratch2)
  hgemm(handle, M, d_ff, d, x, w.W_up,   scratch2);
  // Down: [M, d_ff] x [d_ff, d] (reading from scratch1, writing to x)
  hgemm(handle, M, d, d_ff, scratch1, w.W_down, x);
}

void forward_model(
  cublasHandle_t handle,
  const ModelConfig& cfg,
  LayerWeights* all_weights,
  int M,
  half* x,
  half* scratch1,
  half* scratch2
) {
  for (int l = 0; l < cfg.n_layers; l++)
    forward_layer(handle, cfg, all_weights[l], M, x, scratch1, scratch2);
}

// ---------------------------------------------------------------------------
// Activation buffer
// ---------------------------------------------------------------------------
static int max_activation_dim(const ModelConfig& cfg) {
  return (cfg.q_dim() > cfg.d_ff) ? cfg.q_dim() : cfg.d_ff;
}

ActivationBuffers alloc_activations(const ModelConfig& cfg, int M) {
  ActivationBuffers buf;
  int max_dim = max_activation_dim(cfg);
  CHECK_CUDA(cudaMalloc(
    &buf.x, static_cast<size_t>(M) * cfg.d_model * sizeof(half)
  ));
  CHECK_CUDA(cudaMalloc(
    &buf.scratch1, static_cast<size_t>(M) * max_dim * sizeof(half)
  ));
  CHECK_CUDA(cudaMalloc(
    &buf.scratch2, static_cast<size_t>(M) * max_dim * sizeof(half)
  ));
  CHECK_CUDA(cudaMemset(
    buf.x, 0x3C, static_cast<size_t>(M) * cfg.d_model * sizeof(half)
  ));
  return buf;
}

void free_activations(ActivationBuffers& buf) {
  cudaFree(buf.x);
  cudaFree(buf.scratch1);
  cudaFree(buf.scratch2);
}
