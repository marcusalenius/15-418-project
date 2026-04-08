// ar_decode.cu
// Single-GPU cuBLAS GEMM benchmark for full AR decode throughput.
// Generating N tokens through N sequential full forward passes (each L layers).
// Models: Llama-3.1-70B (target) and Llama-3.2-1B (draft)
//
// Compile: nvcc -o ar_decode ar_decode.cu -lcublas -O2 -ccbin /usr/bin/g++-11
// Run:     ./ar_decode

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
// Model config
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
// Allocate weight matrices for one layer, fill with ~1.0
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
  const LayerWeights& w,
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
  hgemm(handle, M, d_q,  d, x, w.W_q, scratch1);
  // K: [M, d] x [d, d_kv] (writing to scratch2)
  hgemm(handle, M, d_kv, d, x, w.W_k, scratch2);
  // V: [M, d] x [d, d_kv] (writing to scratch2)
  hgemm(handle, M, d_kv, d, x, w.W_v, scratch2);
  // O: [M, d_q] x [d_q, d] (reading from scratch1, writing to x)
  hgemm(handle, M, d, d_q, scratch1, w.W_o, x);

  // --- FFN projections ---
  // Gate: [M, d] x [d, d_ffn] (writing to scratch1)
  hgemm(handle, M, d_ffn, d, x, w.W_gate, scratch1);
  // Up: [M, d] x [d, d_ffn] (writing to scratch2)
  hgemm(handle, M, d_ffn, d, x, w.W_up,   scratch2);
  // Down: [M, d_ffn] x [d_ffn, d] (reading from scratch1, writing to x)
  hgemm(handle, M, d, d_ffn, scratch1, w.W_down, x);
}

// ---------------------------------------------------------------------------
// Full forward pass, i.e. L layers sequentially
// ---------------------------------------------------------------------------
static void forward_full_model(
  cublasHandle_t handle, 
  const ModelConfig& cfg,
  LayerWeights* all_weights,
  int M, 
  half* x, 
  half* scratch1, 
  half* scratch2
) {
  for (int l = 0; l < cfg.n_layers; l++)
    forward_layer_gemms(handle, cfg, all_weights[l], M, x, scratch1, scratch2);
}

// ---------------------------------------------------------------------------
// Generate N tokens: run the full L-layer model N times (AR decode).
// Returns total ms to generate N tokens (averaged over bench_iters runs).
// ---------------------------------------------------------------------------
static float benchmark_ar_decode(
  cublasHandle_t handle, 
  const ModelConfig& cfg,
  int N_tokens, 
  int M,
  int warmup_iters, 
  int bench_iters
) {
  int L = cfg.n_layers;
  LayerWeights* all_weights = new LayerWeights[L];
  for (int l = 0; l < L; l++) alloc_layer_weights(&all_weights[l], cfg);

  // Allocate activation buffers
  int max_act = (cfg.q_dim() > cfg.d_ffn) ? cfg.q_dim() : cfg.d_ffn;
  half *x, *s1, *s2;
  CHECK_CUDA(cudaMalloc(&x,  (size_t)M * cfg.d_model * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&s1, (size_t)M * max_act * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&s2, (size_t)M * max_act * sizeof(half)));
  CHECK_CUDA(cudaMemset(x, 0x3C, (size_t)M * cfg.d_model * sizeof(half)));

  // Warmup
  for (int i = 0; i < warmup_iters; i++)
    forward_full_model(handle, cfg, all_weights, M, x, s1, s2);
  CHECK_CUDA(cudaDeviceSynchronize());

  // Timed: generate N_tokens, averaged over bench_iters trials
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < bench_iters; i++) {
    for (int t = 0; t < N_tokens; t++) {
      forward_full_model(handle, cfg, all_weights, M, x, s1, s2);
    }
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float total_ms = 0;
  CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
  float per_generation_ms = total_ms / bench_iters;

  // Cleanup
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  cudaFree(x); cudaFree(s1); cudaFree(s2);
  for (int l = 0; l < L; l++) free_layer_weights(&all_weights[l]);
  delete[] all_weights;
  return per_generation_ms;
}


// ---------------------------------------------------------------------------
// Single-layer benchmark (fallback when full model doesn't fit).
// Returns total ms to compute one layer (averaged over bench_iters runs).
// ---------------------------------------------------------------------------
static float benchmark_layer(
  cublasHandle_t handle,
  const ModelConfig& cfg,
  int M,
  int warmup_iters,
  int bench_iters
) {
  LayerWeights weights;
  alloc_layer_weights(&weights, cfg);

  // Allocate activation buffers
  int max_act = (cfg.q_dim() > cfg.d_ffn) ? cfg.q_dim() : cfg.d_ffn;
  half *x, *s1, *s2;
  CHECK_CUDA(cudaMalloc(&x,  (size_t)M * cfg.d_model * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&s1, (size_t)M * max_act * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&s2, (size_t)M * max_act * sizeof(half)));
  CHECK_CUDA(cudaMemset(x, 0x3C, (size_t)M * cfg.d_model * sizeof(half)));

  // Warmup
  for (int i = 0; i < warmup_iters; i++) {
    forward_layer_gemms(handle, cfg, weights, M, x, s1, s2);
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  // Timed
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < bench_iters; i++) {
    forward_layer_gemms(handle, cfg, weights, M, x, s1, s2);
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
  cudaFree(s1);
  cudaFree(s2);
  free_layer_weights(&weights);

  return per_layer_ms;
}

// ---------------------------------------------------------------------------
// Utility functins for roofline analysis
// ---------------------------------------------------------------------------

// Total weight bytes per layer
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

// Arithmetic intensity of one layer
static double compute_arith_intensity(const ModelConfig& cfg, int M) {
  int d = cfg.d_model;
  int d_q = cfg.q_dim();
  int d_kv = cfg.kv_dim();
  int d_ffn = cfg.d_ffn;
  double flops = 0;
  double bytes = 0;
  // Q, K, V, O, gate, up, down
  int gemm_dims[][3] = {
    {M, d_q, d}, {M, d_kv, d}, {M, d_kv, d}, {M, d, d_q},
    {M, d_ffn, d}, {M, d_ffn, d}, {M, d, d_ffn}
  };
  for (auto& g : gemm_dims) {
    int gm = g[0], gn = g[1], gk = g[2];
    flops += 2.0 * gm * gn * gk;
    bytes += ((double)gm * gk + (double)gk * gn + 
              (double)gm * gn) * sizeof(half);
  }
  return flops / bytes;
}

// ---------------------------------------------------------------------------
int main() {
  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));

  // Use tensor cores when available (V100 supports FP16 tensor cores)
  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  printf("=== Full-Model AR Decode Throughput Benchmark ===\n");
  printf("    AR decode: each token is one full forward pass (L layers)\n");
  printf("    Generating N tokens is running the model N times\n\n");

  // Print GPU info
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s\n", prop.name);
  printf("Memory bandwidth: %.0f GB/s\n",
         prop.memoryBusWidth / 8.0 * prop.memoryClockRate * 2.0 / 1e6);
  printf("SM count: %d\n", prop.multiProcessorCount);

  size_t free_mem, total_mem;
  CHECK_CUDA(cudaMemGetInfo(&free_mem, &total_mem));
  printf("GPU memory: %.2f GB free / %.2f GB total\n\n",
         free_mem/(1024.0*1024*1024), total_mem/(1024.0*1024*1024));

  const int WARMUP = 10;
  const int BENCH  = 20;
  const int N_GEN  = 128;  // tokens to generate per measurement

  const ModelConfig* models[] = {&DRAFT, &TARGET};

  for (int mi = 0; mi < 2; mi++) {
    const ModelConfig& cfg = *models[mi];
    size_t wb = layer_weight_bytes(cfg);
    size_t total_wb = wb * cfg.n_layers;
    bool fits = total_wb < free_mem * 0.85;

    printf("--- %s ---\n", cfg.name);
    printf("  d_model=%d, n_heads=%d, n_kv_heads=%d, head_dim=%d, d_ffn=%d, layers=%d\n",
           cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.d_ffn, cfg.n_layers);
    printf("  Per-layer weights: %.2f MB | Total: %.2f GB\n",
           wb/(1024.0*1024), total_wb/(1024.0*1024*1024));
    printf("  Weights fit in GPU: %s\n\n", fits ? "YES (full model)" : "NO (layer*L estimate)");

    // --- AR decode: generate N tokens, M=1 each ---
    printf("  [AR Decode] Generate %d tokens (M=1 per step, %d layers per step)\n", N_GEN, cfg.n_layers);
    if (fits) {
      float gen_ms = benchmark_ar_decode(handle, cfg, N_GEN, 1, WARMUP, BENCH);
      float per_tok = gen_ms / N_GEN;
      printf("    Total: %.2f ms | Per-token: %.3f ms | Throughput: %.1f tok/s\n",
             gen_ms, per_tok, N_GEN * 1000.0f / gen_ms);
    } else {
      float layer_ms = benchmark_layer(handle, cfg, 1, WARMUP, BENCH*5);
      float per_tok = layer_ms * cfg.n_layers;
      printf("    Per-token: %.3f ms (layer %.3f ms x %d) | Throughput: %.1f tok/s (est)\n",
             per_tok, layer_ms, cfg.n_layers, 1000.0f / per_tok);
    }
    printf("    Arith intensity (M=1): %.2f\n\n", compute_arith_intensity(cfg, 1));

    // --- Verification pass: M tokens in one forward pass ---
    printf("  [Verification] One forward pass with M tokens batched\n");
    printf("    %5s  %14s  %14s  %12s\n", "M", "pass (ms)", "throughput", "arith intens");
    printf("    %5s  %14s  %14s  %12s\n", "-----", "--------------", "--------------", "------------");
    for (int M : {2, 4, 5, 8, 16}) {
      if (fits) {
        float pass_ms = benchmark_ar_decode(handle, cfg, 1, M, WARMUP, BENCH);
        printf("    %5d  %12.3f ms  %10.1f tok/s  %10.2f\n",
               M, pass_ms, M*1000.0f/pass_ms, compute_arith_intensity(cfg, M));
      } else {
        float lms = benchmark_layer(handle, cfg, M, WARMUP, BENCH*5);
        float pass_ms = lms * cfg.n_layers;
        printf("    %5d  %12.3f ms  %10.1f tok/s  %10.2f  (est)\n",
               M, pass_ms, M*1000.0f/pass_ms, compute_arith_intensity(cfg, M));
      }
    }
    printf("\n");
  }

  // SD cost model
  printf("=== Speculative Decoding Cost Model ===\n");
  printf("  One SD round = K draft AR steps + 1 target verify (M=K)\n\n");

  size_t draft_wb = layer_weight_bytes(DRAFT) * DRAFT.n_layers;
  bool draft_fits = draft_wb < free_mem * 0.85;

  float draft_per_tok;
  if (draft_fits) {
    float gen_ms = benchmark_ar_decode(handle, DRAFT, N_GEN, 1, WARMUP, BENCH);
    draft_per_tok = gen_ms / N_GEN;
    printf("  Draft per-token: %.3f ms (stacked, %d layers)\n", draft_per_tok, DRAFT.n_layers);
  } else {
    float lms = benchmark_layer(handle, DRAFT, 1, WARMUP, BENCH*5);
    draft_per_tok = lms * DRAFT.n_layers;
    printf("  Draft per-token: %.3f ms (est, %d layers)\n", draft_per_tok, DRAFT.n_layers);
  }

  size_t target_wb = layer_weight_bytes(TARGET) * TARGET.n_layers;
  bool target_fits = target_wb < free_mem * 0.85;

  float target_per_tok;
  if (target_fits) {
    float gen_ms = benchmark_ar_decode(handle, TARGET, N_GEN, 1, WARMUP, BENCH);
    target_per_tok = gen_ms / N_GEN;
    printf("  Target per-token: %.3f ms (stacked, %d layers) = %.1f tok/s AR baseline\n\n",
           target_per_tok, TARGET.n_layers, 1000.0f / target_per_tok);
  } else {
    float lms = benchmark_layer(handle, TARGET, 1, WARMUP, BENCH*5);
    target_per_tok = lms * TARGET.n_layers;
    printf("  Target per-token: %.3f ms (est, %d layers) = %.1f tok/s AR baseline\n\n",
           target_per_tok, TARGET.n_layers, 1000.0f / target_per_tok);
  }

  printf("  %3s  %12s  %12s  %12s  %8s  %12s\n",
         "K", "draft (ms)", "verify (ms)", "round (ms)", "T_SD", "eff tok/s*");
  printf("  %3s  %12s  %12s  %12s  %8s  %12s\n",
         "---", "------------", "------------", "------------", "--------", "------------");

  for (int K = 1; K <= 8; K++) {
    float draft_ms = K * draft_per_tok;

    float verify_ms;
    if (target_fits) {
      verify_ms = benchmark_ar_decode(handle, TARGET, 1, K, WARMUP, BENCH);
    } else {
      float tgt_layer_K = benchmark_layer(handle, TARGET, K, WARMUP, BENCH*5);
      verify_ms = tgt_layer_K * TARGET.n_layers;
    }
    float round_ms = draft_ms + verify_ms;
    float T_SD = draft_ms / verify_ms;
    float eff_toks = (K + 1) * 1000.0f / round_ms;

    printf("  %3d  %10.2f ms  %10.2f ms  %10.2f ms  %8.4f  %10.1f\n",
           K, draft_ms, verify_ms, round_ms, T_SD, eff_toks);
  }
  printf("\n  * eff tok/s assumes 100%% acceptance rate (upper bound)\n");

  CHECK_CUBLAS(cublasDestroy(handle));
  return 0;
}
