// transformer_layer.cu
// Single-GPU cuBLAS GEMM sequence for one transformer layer.
// Models: Llama-3.1-70B (target) and Llama-3.2-1B (draft)
//
// Compile: nvcc -o transformer_layer transformer_layer.cu -lcublas -O2 -ccbin /usr/bin/g++-11
// Run:     ./transformer_layer

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CHECK_CUBLAS(call)                                                     \
  do {                                                                         \
    cublasStatus_t status = (call);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__,       \
              status);                                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------------------
// Model config for one decoder layer
// ---------------------------------------------------------------------------
struct ModelConfig {
  const char* name;
  int d_model;       // hidden dimension
  int n_heads;       // number of Q attention heads
  int n_kv_heads;    // number of KV attention heads (GQA)
  int head_dim;      // per-head dimension
  int d_ffn;         // feed-forward intermediate dimension
  int n_layers;      // total layers in the model

  int q_dim() const { return n_heads * head_dim; }    // = d_model for Llama
  int kv_dim() const { return n_kv_heads * head_dim; }
};

// Llama-3.1-70B-Instruct (target)
static const ModelConfig TARGET = {
    .name = "Llama-3.1-70B (target)",
    .d_model = 8192,
    .n_heads = 64,
    .n_kv_heads = 8,
    .head_dim = 128,
    .d_ffn = 28672,
    .n_layers = 80,
};

// Llama-3.2-1B-Instruct (draft)
static const ModelConfig DRAFT = {
    .name = "Llama-3.2-1B (draft)",
    .d_model = 2048,
    .n_heads = 32,
    .n_kv_heads = 8,
    .head_dim = 64,
    .d_ffn = 8192,
    .n_layers = 16,
};

// ---------------------------------------------------------------------------
// Weight buffers for one layer (FP16)
// ---------------------------------------------------------------------------
struct LayerWeights {
  half* W_q;     // [d_model, q_dim]
  half* W_k;     // [d_model, kv_dim]
  half* W_v;     // [d_model, kv_dim]
  half* W_o;     // [d_model, d_model]   (q_dim == d_model for Llama)
  half* W_gate;  // [d_model, d_ffn]
  half* W_up;    // [d_model, d_ffn]
  half* W_down;  // [d_ffn, d_model]
};

// ---------------------------------------------------------------------------
// Allocate weight matrices for one layer, fill with random bits
// ---------------------------------------------------------------------------
static size_t alloc_layer_weights(LayerWeights* w, const ModelConfig& cfg) {
  size_t total = 0;
  auto alloc = [&](half** ptr, int rows, int cols) {
    size_t bytes = (size_t)rows * cols * sizeof(half);
    CHECK_CUDA(cudaMalloc(ptr, bytes));
    // Fill with arbitrary nonzero data so tensor cores aren't shortcutted
    CHECK_CUDA(cudaMemset(*ptr, 0x3C, bytes)); // ~1.0 in FP16
    total += bytes;
  };

  alloc(&w->W_q,    cfg.d_model, cfg.q_dim());
  alloc(&w->W_k,    cfg.d_model, cfg.kv_dim());
  alloc(&w->W_v,    cfg.d_model, cfg.kv_dim());
  alloc(&w->W_o,    cfg.d_model, cfg.d_model);
  alloc(&w->W_gate, cfg.d_model, cfg.d_ffn);
  alloc(&w->W_up,   cfg.d_model, cfg.d_ffn);
  alloc(&w->W_down, cfg.d_ffn,   cfg.d_model);
  return total;
}

static void free_layer_weights(LayerWeights* w) {
  cudaFree(w->W_q);
  cudaFree(w->W_k);
  cudaFree(w->W_v);
  cudaFree(w->W_o);
  cudaFree(w->W_gate);
  cudaFree(w->W_up);
  cudaFree(w->W_down);
}

// ---------------------------------------------------------------------------
// Run one GEMM: C = A * B
// ---------------------------------------------------------------------------

// Given A [MxK], B [KxN], and C [MxN], computes C = A @ B [MxN].
// cuBLAS assumes column-major layout. So, it sees A^T [KxM], B^T [NxK],
// and C^T [NxM], all column-major. And computes C^T = B^T @ A^T [NxM],
// which is equal to C = A @ B [MxN].
static void hgemm(
  cublasHandle_t handle,
  int M, int N, int K,
  const half* A, const half* B, half* C
) {
  const half alpha = __float2half(1.0f);
  const half beta  = __float2half(0.0f);

  CHECK_CUBLAS(
    // Signature: 
    //   (handle, 
    //    opA, opB, 
    //    m, n, k, 
    //    alpha, 
    //    A, lda, 
    //    B, ldb, 
    //    beta, 
    //    C, ldc)
    // Notes:
    //  - m = N, n = M, k = K -- the dimensions of the column-major C^T
    //  - opA = opB = CUBLAS_OP_N -- no transpose
    //  - lda = leading dimension A, etc for ldb and ldc
    cublasHgemm(
      handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      N, M, K,
      &alpha,
      B, N,
      A, K,
      &beta,
      C, N
    )
  );     
}

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
// ---------------------------------------------------------------------------
static void forward_layer_gemms(
  cublasHandle_t handle,
  const ModelConfig& cfg,
  const LayerWeights& weights,
  int M,           // number of tokens (1 for AR, K for verification)
  half* x,         // [M, d] input activations
  half* scratch1,  // workspace >= [M, max(d_q, d_ffn)]
  half* scratch2   // workspace >= [M, max(d_kv, d_ffn)]
) {
  int d = cfg.d_model;
  int d_q = cfg.q_dim();
  int d_kv = cfg.kv_dim();
  int d_ffn = cfg.d_ffn;

  // --- Attention projections ---
  // Q: [M, d] x [d, d_q] (writing to scratch1)
  hgemm(handle, M, d_q,  d, x, weights.W_q, scratch1);
  // K: [M, d] x [d, d_kv] (writing to scratch2)
  hgemm(handle, M, d_kv, d, x, weights.W_k, scratch2);
  // V: [M, d] x [d, d_kv] (writing to scratch2)
  hgemm(handle, M, d_kv, d, x, weights.W_v, scratch2);
  // O: [M, d_q] x [d_q, d] (reading from scratch1, writing to x)
  hgemm(handle, M, d, d_q, scratch1, weights.W_o, x);

  // --- FFN projections ---
  // Gate: [M, d] x [d, d_ffn] (writing to scratch1)
  hgemm(handle, M, d_ffn, d, x, weights.W_gate, scratch1);
  // Up: [M, d] x [d, d_ffn] (writing to scratch2)
  hgemm(handle, M, d_ffn, d, x, weights.W_up,   scratch2);
  // Down: [M, d_ffn] x [d_ffn, d] (reading from scratch1, writing to x)
  hgemm(handle, M, d, d_ffn, scratch1, weights.W_down, x);
}

// ---------------------------------------------------------------------------
// Benchmark one layer for a given model config and token count M
// ---------------------------------------------------------------------------
static float benchmark_layer(
  cublasHandle_t handle,
  const ModelConfig& cfg,
  int M,
  int warmup_iters,
  int bench_iters
) {
  LayerWeights weights;
  size_t weight_bytes = alloc_layer_weights(&weights, cfg);

  // Allocate activation buffers
  int max_act = (cfg.q_dim() > cfg.d_ffn) ? cfg.q_dim() : cfg.d_ffn;
  size_t x_bytes = (size_t)M * cfg.d_model * sizeof(half);
  size_t s_bytes = (size_t)M * max_act * sizeof(half);

  half *x, *scratch1, *scratch2;
  CHECK_CUDA(cudaMalloc(&x, x_bytes));
  CHECK_CUDA(cudaMalloc(&scratch1, s_bytes));
  CHECK_CUDA(cudaMalloc(&scratch2, s_bytes));
  CHECK_CUDA(cudaMemset(x, 0x3C, x_bytes));

  // Warmup
  for (int i = 0; i < warmup_iters; i++) {
    forward_layer_gemms(handle, cfg, weights, M, x, scratch1, scratch2);
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  // Timed runs
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < bench_iters; i++) {
    forward_layer_gemms(handle, cfg, weights, M, x, scratch1, scratch2);
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float total_ms = 0;
  CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
  float per_layer_ms = total_ms / bench_iters;

  // Cleanup
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  cudaFree(x);
  cudaFree(scratch1);
  cudaFree(scratch2);
  free_layer_weights(&weights);

  return per_layer_ms;
}

// ---------------------------------------------------------------------------
// Compute total weight bytes per layer (for roofline analysis)
// ---------------------------------------------------------------------------
static size_t layer_weight_bytes(const ModelConfig& cfg) {
  size_t b = sizeof(half);
  int d = cfg.d_model;
  int d_q = cfg.q_dim();
  int d_kv = cfg.kv_dim();
  int d_ffn = cfg.d_ffn;
  return b * (
    (size_t)d * d_q +
    (size_t)d * d_kv +
    (size_t)d * d_kv +
    (size_t)d_q * d +
    (size_t)d * d_ffn +
    (size_t)d * d_ffn +
    (size_t)d_ffn * d
  );
}

// ---------------------------------------------------------------------------
int main() {
  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));

  // Use tensor cores when available (V100 supports FP16 tensor cores)
  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  printf("=== Transformer Layer GEMM Benchmark ===\n\n");

  // Print GPU info
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s\n", prop.name);
  printf("Memory bandwidth: %.0f GB/s\n",
         prop.memoryBusWidth / 8.0 * prop.memoryClockRate * 2.0 / 1e6);
  printf("SM count: %d\n\n", prop.multiProcessorCount);

  const int WARMUP = 20;
  const int BENCH  = 100;

  // Token counts to test: M=1 (AR decode), M=K for verification
  int token_counts[] = {1, 2, 4, 5, 8, 16};
  int n_tokens = sizeof(token_counts) / sizeof(token_counts[0]);

  const ModelConfig* models[] = {&DRAFT, &TARGET};

  for (int mi = 0; mi < 2; mi++) {
    const ModelConfig& cfg = *models[mi];
    size_t wb = layer_weight_bytes(cfg);

    printf("--- %s ---\n", cfg.name);
    printf("  d_model=%d, n_heads=%d, n_kv_heads=%d, head_dim=%d, d_ffn=%d\n",
           cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.d_ffn);
    printf("  Layer weight size: %.2f MB\n", wb / (1024.0 * 1024.0));
    printf("  Total model weights (%d layers): %.2f GB\n\n",
           cfg.n_layers, wb * cfg.n_layers / (1024.0 * 1024.0 * 1024.0));

    printf("  %5s  %12s  %12s  %12s\n",
           "M", "layer (ms)", "full model*", "arith intens");
    printf("  %5s  %12s  %12s  %12s\n",
           "-----", "------------", "------------", "------------");

    for (int ti = 0; ti < n_tokens; ti++) {
      int M = token_counts[ti];

      float layer_ms = benchmark_layer(handle, cfg, M, WARMUP, BENCH);
      float full_model_ms = layer_ms * cfg.n_layers;  // estimate

      // Arithmetic intensity: FLOPs / bytes_read
      // For each GEMM (M,N,K): 2*M*N*K FLOPs, reads M*K + K*N elements
      double total_flops = 0;
      double total_bytes = 0;
      int d = cfg.d_model, d_q = cfg.q_dim(), 
          d_kv = cfg.kv_dim(), d_ffn = cfg.d_ffn;
      // Q, K, V, O, gate, up, down
      int gemm_dims[][3] = {
        {M, d_q, d}, {M, d_kv, d}, {M, d_kv, d}, {M, d, d_q},
        {M, d_ffn, d}, {M, d_ffn, d}, {M, d, d_ffn}
      };
      for (auto& g : gemm_dims) {
        int gm = g[0], gn = g[1], gk = g[2];
        total_flops += 2.0 * gm * gn * gk;
        total_bytes += ((double)gm * gk + (double)gk * gn + 
                        (double)gm * gn) * sizeof(half);
      }
      double arith_intensity = total_flops / total_bytes;

      printf("  %5d  %10.3f ms  %10.3f ms  %10.2f\n",
             M, layer_ms, full_model_ms, arith_intensity);
    }
    printf("\n  * Full model estimate = single-layer time x %d layers (no attention/norm)\n\n",
           cfg.n_layers);
  }

  // Summary: draft-to-target time ratio T_SD
  printf("=== Draft-to-Target Time Ratio (T_SD) ===\n");
  printf("  T_SD = (K * T_draft_layer * L_draft) / (T_target_layer * L_target)\n");
  printf("  (Measure from the table above for your K of interest)\n\n");

  float t_draft_1  = benchmark_layer(handle, DRAFT,  1, WARMUP, BENCH);
  float t_target_1 = benchmark_layer(handle, TARGET, 1, WARMUP, BENCH);

  for (int K = 1; K <= 8; K++) {
    // SD: K sequential draft passes + 1 target verification (with K tokens)
    float t_target_K = benchmark_layer(handle, TARGET, K, WARMUP, BENCH);
    float t_draft_total = K * t_draft_1 * DRAFT.n_layers;
    float t_target_total = t_target_K * TARGET.n_layers;
    float T_SD = t_draft_total / t_target_total;

    printf("  K=%d: T_draft_total=%.2f ms, T_target_verify=%.2f ms, T_SD=%.3f\n",
           K, t_draft_total, t_target_total, T_SD);
  }

  CHECK_CUBLAS(cublasDestroy(handle));
  return 0;
}
