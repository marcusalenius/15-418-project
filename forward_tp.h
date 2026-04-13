// forward_tp.h
// Multi-GPU tensor-parallel forward pass functions.

#pragma once

#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <nccl.h>
#include "config.h"
#include "forward_single.h"  // ActivationBuffers

// ---------------------------------------------------------------------------
// Tensor-parallel weight buffers for one layer on one GPU.
//
// Sharding strategy (Megatron-style):
//   Column-parallel: W_q, W_k, W_v, W_gate, W_up  -> shard output dim
//     Each GPU stores [d_model, dim/T] slice
//   Row-parallel:    W_o, W_down                  -> shard input dim
//     Each GPU stores [dim/T, d_model] slice
//     Output is partial sum -> needs all-reduce
// ---------------------------------------------------------------------------
struct TPLayerWeights {
  half* W_q;     // [d_model, q_dim/T]     column-parallel
  half* W_k;     // [d_model, kv_dim/T]    column-parallel
  half* W_v;     // [d_model, kv_dim/T]    column-parallel
  half* W_o;     // [q_dim/T, d_model]     row-parallel
  half* W_gate;  // [d_model, d_ffn/T]     column-parallel
  half* W_up;    // [d_model, d_ffn/T]     column-parallel
  half* W_down;  // [d_ffn/T, d_model]     row-parallel
};

size_t alloc_tp_layer_weights(TPLayerWeights& w, const ModelConfig& cfg, int T);
void free_tp_layer_weights(TPLayerWeights& w);

// ---------------------------------------------------------------------------
// Forward passes
// ---------------------------------------------------------------------------

// One layer: 7 GEMMs + 2 all-reduces (after O and after down)
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
);

// Full model: L layers sequentially
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
);

// ---------------------------------------------------------------------------
// Activation buffers (TP-aware sizing)
// ---------------------------------------------------------------------------
ActivationBuffers alloc_activations_tp(const ModelConfig& cfg, int M, int T);
