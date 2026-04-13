// forward_single.h
// Single-GPU forward pass and benchmarking functions.

#pragma once

#include "config.h"

// ---------------------------------------------------------------------------
// Weight buffers for one layer (FP16, single GPU)
// ---------------------------------------------------------------------------
struct LayerWeights {
  half* W_q;     // [d_model, q_dim]
  half* W_k;     // [d_model, kv_dim]
  half* W_v;     // [d_model, kv_dim]
  half* W_o;     // [q_dim,   d_model]
  half* W_gate;  // [d_model, d_ffn]
  half* W_up;    // [d_model, d_ffn]
  half* W_down;  // [d_ffn,   d_model]
};

// Allocate and fill weight matrices for one layer. Returns total bytes.
size_t alloc_layer_weights(LayerWeights& w, const ModelConfig& cfg);
void free_layer_weights(LayerWeights& w);

// ---------------------------------------------------------------------------
// Forward passes
// ---------------------------------------------------------------------------

// Forward pass for one layer (GEMMs only)
//
// Sequence of ops in a real Llama layer:
//   1. RMSNorm (skip -- negligible time)
//   2. Q = x @ W_q      [M, d] x [d, d_q]          -> [M, d_q]
//   3. K = x @ W_k      [M, d] x [d, d_kv]         -> [M, d_kv]
//   4. V = x @ W_v      [M, d] x [d, d_kv]         -> [M, d_kv]
//   5. Attention (skip for now -- separate kernel, not a GEMM)
//   6. O = attn @ W_o   [M, d_q] x [d_q, d]        -> [M, d]  (d_q == d)
//   7. RMSNorm (skip)
//   8. gate = x @ W_gate [M, d] x [d, d_ffn]       -> [M, d_ffn]
//   9. up   = x @ W_up   [M, d] x [d, d_ffn]       -> [M, d_ffn]
//  10. SiLU(gate) * up   (skip -- element-wise)
//  11. down = act @ W_down [M, d_ffn] x [d_ffn, d] -> [M, d]
//
// Total: 7 GEMMs per layer.
void forward_layer(
  cublasHandle_t handle,
  const ModelConfig& cfg,
  const LayerWeights& w,
  int M,           // number of tokens (1 for AR, K for verification)
  half* x,         // [M, d] input activations
  half* scratch1,  // workspace >= [M, max(d_q, d_ffn)]
  half* scratch2   // workspace >= [M, max(d_kv, d_ffn)]
);

// Full model: L layers sequentially
void forward_model(
  cublasHandle_t handle,
  const ModelConfig& cfg,
  LayerWeights* all_weights,
  int M,
  half* x,
  half* scratch1,
  half* scratch2
);

// ---------------------------------------------------------------------------
// Activation buffers
// ---------------------------------------------------------------------------
struct ActivationBuffers {
  half* x;
  half* scratch1;
  half* scratch2;
};

ActivationBuffers alloc_activations(const ModelConfig& cfg, int M);
void free_activations(ActivationBuffers& buf);
