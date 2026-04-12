// ar_decode_tp.cu
// Multi-GPU tensor-parallel AR decode benchmark.
// Shards weight matrices across T GPUs with NCCL all-reduce after each layer.
//
// Compile: nvcc -o ar_decode_tp ar_decode_tp.cu -lcublas -lnccl -O2 -ccbin /usr/bin/g++-11
// Run:     ./ar_decode_tp T  (spawns T threads, one per GPU)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <nccl.h>

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

#define CHECK_NCCL(call)                                                       \
  do {                                                                         \
    ncclResult_t res = (call);                                                 \
    if (res != ncclSuccess) {                                                  \
      fprintf(stderr, "NCCL error at %s:%d: %s\n", __FILE__, __LINE__,         \
              ncclGetErrorString(res));                                        \
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
  half* W_q;     // [d_model, q_dim/T]       column-parallel
  half* W_k;     // [d_model, kv_dim/T]      column-parallel
  half* W_v;     // [d_model, kv_dim/T]      column-parallel
  half* W_o;     // [q_dim/T, d_model]       row-parallel
  half* W_gate;  // [d_model, d_ffn/T]       column-parallel
  half* W_up;    // [d_model, d_ffn/T]       row-parallel (same shape as gate)
  half* W_down;  // [d_ffn/T, d_model]       row-parallel
};

static size_t alloc_tp_layer_weights(
  TPLayerWeights* w, 
  const ModelConfig& cfg, 
  int T
) {
  size_t total = 0;
  auto alloc = [&](half** ptr, int rows, int cols) {
    size_t bytes = (size_t)rows * cols * sizeof(half);
    CHECK_CUDA(cudaMalloc(ptr, bytes));
    // Fill with arbitrary nonzero data so tensor cores aren't shortcutted
    CHECK_CUDA(cudaMemset(*ptr, 0x3C, bytes)); // ~1.0 in FP16
    total += bytes;
  };

  int d = cfg.d_model;
  int d_q = cfg.q_dim();
  int d_kv = cfg.kv_dim();
  int d_ffn = cfg.d_ffn;

  // Column-parallel: shard output dimension
  alloc(&w->W_q,    d, d_q / T);
  alloc(&w->W_k,    d, d_kv / T);
  alloc(&w->W_v,    d, d_kv / T);
  // Row-parallel: shard input dimension
  alloc(&w->W_o,    d_q / T, d);
  // Column-parallel
  alloc(&w->W_gate, d, d_ffn / T);
  alloc(&w->W_up,   d, d_ffn / T);
  // Row-parallel
  alloc(&w->W_down, d_ffn / T, d);

  return total;
}

static void free_tp_layer_weights(TPLayerWeights* w) {
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
// Tensor-parallel forward pass for one layer (GEMMs only).
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
//
// Per layer, each GPU does:
//   Column-parallel GEMMs (Q, K, V, gate, up): full input, 1/T output
//   Row-parallel GEMMs (O, down): 1/T input, full output (partial sum)
//   All-reduce after O and after down to sum partial results
// ---------------------------------------------------------------------------
static void forward_layer_tp(
  cublasHandle_t handle,
  ncclComm_t comm,
  cudaStream_t stream,
  const ModelConfig& cfg,
  const TPLayerWeights& w,
  int T,             // tensor parallelism degree
  int M,             // batch size (tokens)
  half* x,           // [M, d]  input activations (full)
  half* scratch1,    // workspace
  half* scratch2     // workspace
) {
  int d = cfg.d_model;
  int d_q_local = cfg.q_dim() / T;
  int d_kv_local = cfg.kv_dim() / T;
  int d_ffn_local = cfg.d_ffn / T;

  // Attention projections (column-parallel)
  // Q: [M, d] x [d, d_q/T]  -> [M, d_q/T]     (writing to scratch1)
  hgemm(handle, M, d_q_local, d, x, w.W_q, scratch1);
  // K: [M, d] x [d, d_kv/T] -> [M, d_kv/T]    (writing to scratch2)
  hgemm(handle, M, d_kv_local, d, x, w.W_k, scratch2);
  // V: [M, d] x [d, d_kv/T] -> [M, d_kv/T]    (writing to scratch2)
  hgemm(handle, M, d_kv_local, d, x, w.W_v, scratch2);

  // O (row-parallel) (partial sum)
  // [M, d_q/T] x [d_q/T, d] -> [M, d]  (reading from scratch1, writing to x)
  hgemm(handle, M, d, d_q_local, scratch1, w.W_o, x);

  // All-reduce after attention output projection
  CHECK_NCCL(
    ncclAllReduce(
      x, x, (size_t)M * d,
      ncclFloat16, ncclSum, comm, stream
    )
  );

  // FFN projections (column-parallel)
  // Gate: [M, d] x [d, d_ffn/T] -> [M, d_ffn/T]   (writing to scratch1)
  hgemm(handle, M, d_ffn_local, d, x, w.W_gate, scratch1);
  // Up: [M, d] x [d, d_ffn/T] -> [M, d_ffn/T]     (writing to scratch2)
  hgemm(handle, M, d_ffn_local, d, x, w.W_up, scratch2);

  // Down (row-parallel) (partial sum)
  // [M, d_ffn/T] x [d_ffn/T, d] -> [M, d]  (reading from scratch1, writing to x)
  hgemm(handle, M, d, d_ffn_local, scratch1, w.W_down, x);

  // All-reduce after FFN down projection
  CHECK_NCCL(
    ncclAllReduce(
      x, x, (size_t)M * d,
      ncclFloat16, ncclSum, comm, stream
    )
  );
}

// ---------------------------------------------------------------------------
// Full forward pass, i.e. L layers sequentially
// ---------------------------------------------------------------------------
static void forward_full_model_tp(
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
// Per-GPU worker context
// ---------------------------------------------------------------------------
struct WorkerArgs {
  int rank;
  int world_size;
  ncclUniqueId nccl_id;
  const ModelConfig* cfg;
  int N_tokens;
  int M;
  int warmup_iters;
  int bench_iters;
  // Output
  float result_ms;
};

static void* worker_thread(void* arg) {
  WorkerArgs* args = (WorkerArgs*)arg;
  int rank = args->rank;
  int T = args->world_size;
  const ModelConfig& cfg = *args->cfg;
  int M = args->M;

  CHECK_CUDA(cudaSetDevice(rank));

  // Create NCCL communicator
  ncclComm_t comm;
  CHECK_NCCL(ncclCommInitRank(&comm, T, args->nccl_id, rank));

  // Create cuBLAS handle and stream
  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));
  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));
  CHECK_CUBLAS(cublasSetStream(handle, stream));

  // Allocate weights for all layers (sharded)
  int L = cfg.n_layers;
  TPLayerWeights* all_weights = new TPLayerWeights[L];
  for (int l = 0; l < L; l++)
    alloc_tp_layer_weights(&all_weights[l], cfg, T);

  // Allocate activation buffers
  int max_local = cfg.q_dim() / T;
  if (cfg.d_ffn / T > max_local) max_local = cfg.d_ffn / T;
  half *x, *s1, *s2;
  CHECK_CUDA(cudaMalloc(&x,  (size_t)M * cfg.d_model * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&s1, (size_t)M * max_local * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&s2, (size_t)M * max_local * sizeof(half)));
  CHECK_CUDA(cudaMemset(x, 0x3C, (size_t)M * cfg.d_model * sizeof(half)));

  // Warmup
  for (int i = 0; i < args->warmup_iters; i++)
    forward_full_model_tp(
      handle, comm, stream, cfg, all_weights, T, M, x, s1, s2
    );
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // Timed: generate N_tokens, averaged over bench_iters trials
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start, stream));
  for (int i = 0; i < args->bench_iters; i++) {
    for (int t = 0; t < args->N_tokens; t++) {
      forward_full_model_tp(
        handle, comm, stream, cfg, all_weights, T, M, x, s1, s2
      );
    }
  }
  CHECK_CUDA(cudaEventRecord(stop, stream));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float total_ms = 0;
  CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
  args->result_ms = total_ms / args->bench_iters;

  // Cleanup
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  cudaFree(x); cudaFree(s1); cudaFree(s2);
  for (int l = 0; l < L; l++) free_tp_layer_weights(&all_weights[l]);
  delete[] all_weights;
  CHECK_CUBLAS(cublasDestroy(handle));
  CHECK_CUDA(cudaStreamDestroy(stream));
  ncclCommDestroy(comm);
  return nullptr;
}

// ---------------------------------------------------------------------------
// Single-layer benchmark (fallback when full model doesn't fit).
// Returns total ms to compute one layer (averaged over bench_iters runs).
// ---------------------------------------------------------------------------
struct LayerWorkerArgs {
  int rank;
  int world_size;
  ncclUniqueId nccl_id;
  const ModelConfig* cfg;
  int M;
  int warmup_iters;
  int bench_iters;
  float result_ms;
};

static void* layer_worker_thread(void* arg) {
  LayerWorkerArgs* args = (LayerWorkerArgs*)arg;
  int rank = args->rank;
  int T = args->world_size;
  const ModelConfig& cfg = *args->cfg;
  int M = args->M;

  CHECK_CUDA(cudaSetDevice(rank));

  ncclComm_t comm;
  CHECK_NCCL(ncclCommInitRank(&comm, T, args->nccl_id, rank));

  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));
  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));
  CHECK_CUBLAS(cublasSetStream(handle, stream));

  TPLayerWeights weights;
  alloc_tp_layer_weights(&weights, cfg, T);

  int max_local = cfg.q_dim() / T;
  if (cfg.d_ffn / T > max_local) max_local = cfg.d_ffn / T;
  half *x, *s1, *s2;
  CHECK_CUDA(cudaMalloc(&x,  (size_t)M * cfg.d_model * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&s1, (size_t)M * max_local * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&s2, (size_t)M * max_local * sizeof(half)));
  CHECK_CUDA(cudaMemset(x, 0x3C, (size_t)M * cfg.d_model * sizeof(half)));

  for (int i = 0; i < args->warmup_iters; i++)
    forward_layer_tp(handle, comm, stream, cfg, weights, T, M, x, s1, s2);
  CHECK_CUDA(cudaStreamSynchronize(stream));

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start, stream));
  for (int i = 0; i < args->bench_iters; i++)
    forward_layer_tp(handle, comm, stream, cfg, weights, T, M, x, s1, s2);
  CHECK_CUDA(cudaEventRecord(stop, stream));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float total_ms = 0;
  CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
  args->result_ms = total_ms / args->bench_iters;

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  cudaFree(x); cudaFree(s1); cudaFree(s2);
  free_tp_layer_weights(&weights);
  CHECK_CUBLAS(cublasDestroy(handle));
  CHECK_CUDA(cudaStreamDestroy(stream));
  ncclCommDestroy(comm);
  return nullptr;
}

// ---------------------------------------------------------------------------
// Launch T threads and return rank-0 timing
// ---------------------------------------------------------------------------
static float run_full_model_benchmark(
  const ModelConfig& cfg, 
  int T, 
  int N_tokens, 
  int M,
  int warmup_iters, 
  int bench_iters
) {
  ncclUniqueId id;
  CHECK_NCCL(ncclGetUniqueId(&id));

  pthread_t* threads = new pthread_t[T];
  WorkerArgs* args = new WorkerArgs[T];

  for (int r = 0; r < T; r++) {
    args[r] = {r, T, id, &cfg, N_tokens, M, warmup_iters, bench_iters, 0.0f};
    pthread_create(&threads[r], nullptr, worker_thread, &args[r]);
  }
  for (int r = 0; r < T; r++)
    pthread_join(threads[r], nullptr);

  float result = args[0].result_ms;
  delete[] threads;
  delete[] args;
  return result;
}

static float run_layer_benchmark(
  const ModelConfig& cfg, 
  int T, 
  int M,
  int warmup_iters, 
  int bench_iters
) {
  ncclUniqueId id;
  CHECK_NCCL(ncclGetUniqueId(&id));

  pthread_t* threads = new pthread_t[T];
  LayerWorkerArgs* args = new LayerWorkerArgs[T];

  for (int r = 0; r < T; r++) {
    args[r] = {r, T, id, &cfg, M, warmup_iters, bench_iters, 0.0f};
    pthread_create(&threads[r], nullptr, layer_worker_thread, &args[r]);
  }
  for (int r = 0; r < T; r++)
    pthread_join(threads[r], nullptr);

  float result = args[0].result_ms;
  delete[] threads;
  delete[] args;
  return result;
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------
static size_t tp_layer_weight_bytes(const ModelConfig& cfg, int T) {
  size_t b = sizeof(half);
  int d = cfg.d_model;
  return b * (
    (size_t)d * (cfg.q_dim() / T) +
    (size_t)d * (cfg.kv_dim() / T) +
    (size_t)d * (cfg.kv_dim() / T) +
    (size_t)(cfg.q_dim() / T) * d +
    (size_t)d * (cfg.d_ffn / T) +
    (size_t)d * (cfg.d_ffn / T) +
    (size_t)(cfg.d_ffn / T) * d
  );
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  int num_gpus;
  CHECK_CUDA(cudaGetDeviceCount(&num_gpus));

  int T = (argc > 1) ? atoi(argv[1]) : num_gpus;
  if (T < 1 || T > num_gpus) {
    fprintf(stderr, "Usage: %s [T]  (T <= %d available GPUs)\n", argv[0], num_gpus);
    return 1;
  }

  printf("=== Tensor-Parallel AR Decode Benchmark ===\n");
  printf("    T = %d GPUs\n\n", T);

  // Print GPU info for each device
  for (int i = 0; i < T; i++) {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, i));
    size_t free_mem, total_mem;
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaMemGetInfo(&free_mem, &total_mem));
    printf("  GPU %d: %s | %.2f GB free / %.2f GB total\n",
           i, prop.name, free_mem/(1024.0*1024*1024), total_mem/(1024.0*1024*1024));
  }
  printf("\n");

  // Check per-GPU memory (use GPU 0 as reference)
  CHECK_CUDA(cudaSetDevice(0));
  size_t free_mem, total_mem;
  CHECK_CUDA(cudaMemGetInfo(&free_mem, &total_mem));

  const int WARMUP = 10;
  const int BENCH  = 20;
  const int N_GEN  = 128;

  const ModelConfig* models[] = {&DRAFT, &TARGET};

  for (int mi = 0; mi < 2; mi++) {
    const ModelConfig& cfg = *models[mi];

    // Validate sharding is evenly divisible
    if (cfg.q_dim() % T != 0 || cfg.kv_dim() % T != 0 || cfg.d_ffn % T != 0) {
      printf("--- %s --- SKIPPED (dimensions not divisible by T=%d)\n\n", cfg.name, T);
      continue;
    }

    size_t per_gpu_wb = tp_layer_weight_bytes(cfg, T) * cfg.n_layers;
    bool fits = per_gpu_wb < free_mem * 0.85;

    printf("--- %s (TP=%d) ---\n", cfg.name, T);
    printf("  Per-GPU weight footprint: %.2f GB | Fits: %s\n",
           per_gpu_wb/(1024.0*1024*1024), fits ? "YES" : "NO (layer*L est)");

    // AR decode M=1
    printf("\n  [AR Decode] Generate %d tokens (M=1)\n", N_GEN);
    if (fits) {
      float gen_ms = run_full_model_benchmark(cfg, T, N_GEN, 1, WARMUP, BENCH);
      float per_tok = gen_ms / N_GEN;
      printf("    Total: %.2f ms | Per-token: %.3f ms | Throughput: %.1f tok/s\n",
             gen_ms, per_tok, N_GEN * 1000.0f / gen_ms);
    } else {
      float lms = run_layer_benchmark(cfg, T, 1, WARMUP, BENCH * 5);
      float per_tok = lms * cfg.n_layers;
      printf("    Per-token: %.3f ms (layer %.3f ms x %d) | Throughput: %.1f tok/s (est)\n",
             per_tok, lms, cfg.n_layers, 1000.0f / per_tok);
    }

    // Verification pass with varying M
    printf("\n  [Verification] One forward pass with M tokens batched\n");
    printf("    %5s  %14s  %14s\n", "M", "pass (ms)", "throughput");
    printf("    %5s  %14s  %14s\n", "-----", "--------------", "--------------");
    for (int M : {2, 4, 5, 8, 16}) {
      if (fits) {
        float pass_ms = run_full_model_benchmark(cfg, T, 1, M, WARMUP, BENCH);
        printf("    %5d  %12.3f ms  %10.1f tok/s\n", M, pass_ms, M * 1000.0f / pass_ms);
      } else {
        float lms = run_layer_benchmark(cfg, T, M, WARMUP, BENCH * 5);
        float pass_ms = lms * cfg.n_layers;
        printf("    %5d  %12.3f ms  %10.1f tok/s  (est)\n", M, pass_ms, M * 1000.0f / pass_ms);
      }
    }
    printf("\n");
  }

  // --- SD Cost Model (same structure as single-GPU) ---
  printf("=== Speculative Decoding Cost Model (TP=%d) ===\n\n", T);

  // Draft per-token
  float draft_per_tok;
  if (DRAFT.q_dim() % T == 0 && DRAFT.kv_dim() % T == 0 && DRAFT.d_ffn % T == 0) {
    size_t dpg = tp_layer_weight_bytes(DRAFT, T) * DRAFT.n_layers;
    if (dpg < free_mem * 0.85) {
      float gen_ms = run_full_model_benchmark(DRAFT, T, N_GEN, 1, WARMUP, BENCH);
      draft_per_tok = gen_ms / N_GEN;
    } else {
      float lms = run_layer_benchmark(DRAFT, T, 1, WARMUP, BENCH * 5);
      draft_per_tok = lms * DRAFT.n_layers;
    }
    printf("  Draft per-token (TP=%d): %.3f ms\n", T, draft_per_tok);
  } else {
    // Draft on single GPU (typical for SSD: draft on separate GPU)
    printf("  Draft dims not divisible by T=%d; would run on single GPU\n", T);
    draft_per_tok = 0; // placeholder
  }

  // Target per-token
  float target_per_tok;
  {
    size_t tpg = tp_layer_weight_bytes(TARGET, T) * TARGET.n_layers;
    bool tf = tpg < free_mem * 0.85;
    if (tf) {
      float gen_ms = run_full_model_benchmark(TARGET, T, N_GEN, 1, WARMUP, BENCH);
      target_per_tok = gen_ms / N_GEN;
    } else {
      float lms = run_layer_benchmark(TARGET, T, 1, WARMUP, BENCH * 5);
      target_per_tok = lms * TARGET.n_layers;
    }
    printf("  Target per-token (TP=%d): %.3f ms = %.1f tok/s AR baseline\n\n",
           T, target_per_tok, 1000.0f / target_per_tok);
  }

  if (draft_per_tok > 0) {
    printf("  %3s  %12s  %12s  %12s  %8s  %12s\n",
           "K", "draft (ms)", "verify (ms)", "round (ms)", "T_SD", "eff tok/s*");
    printf("  %3s  %12s  %12s  %12s  %8s  %12s\n",
           "---", "------------", "------------", "------------", "--------", "------------");

    size_t tpg = tp_layer_weight_bytes(TARGET, T) * TARGET.n_layers;
    bool tf = tpg < free_mem * 0.85;

    for (int K = 1; K <= 8; K++) {
      float draft_ms = K * draft_per_tok;
      float verify_ms;
      if (tf) {
        verify_ms = run_full_model_benchmark(TARGET, T, 1, K, WARMUP, BENCH);
      } else {
        float tgt_lms = run_layer_benchmark(TARGET, T, K, WARMUP, BENCH * 5);
        verify_ms = tgt_lms * TARGET.n_layers;
      }
      float round_ms = draft_ms + verify_ms;
      float T_SD = draft_ms / verify_ms;
      float eff_toks = (K + 1) * 1000.0f / round_ms;
      printf("  %3d  %10.2f ms  %10.2f ms  %10.2f ms  %8.4f  %10.1f\n",
             K, draft_ms, verify_ms, round_ms, T_SD, eff_toks);
    }
    printf("\n  * eff tok/s assumes 100%% acceptance rate (upper bound)\n");
  }

  return 0;
}
