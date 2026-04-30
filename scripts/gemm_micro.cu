// scripts/gemm_micro.cu
// Standalone cuBLAS GEMM microbenchmark for the V100 roofline plot.
// Sweeps over M (token batch) x (model, GEMM kind) and reports per-GEMM:
//   ms, GFLOPs/s, achieved HBM BW (GB/s), arithmetic intensity (FLOP/byte).
//
// Shapes match Llama-{1B, 8B, 70B} QKV / attn_out / FFN-up / FFN-down GEMMs
// in the AR/SD decode forward pass (single-GPU, no TP — TP shapes can be
// derived by dividing the relevant dim by T).
//
// Build (or use scripts/run_microbench.sh which does this):
//   nvcc -O2 -ccbin /usr/bin/g++-11 -std=c++17 -o build/gemm_micro \
//        scripts/gemm_micro.cu -lcublas
//
// Run:
//   ./build/gemm_micro          # human-readable table
//   ./build/gemm_micro --csv    # CSV header + rows
//
// CSV columns:
//   model, kind, M, N, K, ms, gflops, achieved_bw_GBs, arith_intensity

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call) do { \
  cudaError_t e = (call); \
  if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    std::exit(1); \
  } \
} while(0)

#define CHECK_CUBLAS(call) do { \
  cublasStatus_t s = (call); \
  if (s != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)s); \
    std::exit(1); \
  } \
} while(0)

struct GemmShape {
  const char* model;
  const char* kind;   // "qkv", "attn_out", "ffn_up", "ffn_down"
  int N;              // output dim
  int K;              // input dim
};

// Per-layer GEMMs (M = token count, varied separately).
// QKV is fused: N = q_dim + 2 * kv_dim (GQA).
// Llama-3.2-1B:  d=2048, n_heads=32, n_kv=8,  head=64,  ffn=8192
// Llama-3.1-8B:  d=4096, n_heads=32, n_kv=8,  head=128, ffn=14336
// Llama-3.1-70B: d=8192, n_heads=64, n_kv=8,  head=128, ffn=28672
static const std::vector<GemmShape> SHAPES = {
  // Llama-3.2-1B
  {"llama-1b",  "qkv",      2048 + 2*512,  2048},   // q_dim=2048, kv_dim=512
  {"llama-1b",  "attn_out", 2048,          2048},
  {"llama-1b",  "ffn_up",   8192,          2048},   // x2 in real model (gate+up)
  {"llama-1b",  "ffn_down", 2048,          8192},

  // Llama-3.1-8B
  {"llama-8b",  "qkv",      4096 + 2*1024, 4096},   // q_dim=4096, kv_dim=1024
  {"llama-8b",  "attn_out", 4096,          4096},
  {"llama-8b",  "ffn_up",  14336,          4096},
  {"llama-8b",  "ffn_down", 4096,         14336},

  // Llama-3.1-70B (single-GPU shapes; for TP=T divide N by T for col-parallel)
  {"llama-70b", "qkv",      8192 + 2*1024, 8192},   // q_dim=8192, kv_dim=1024
  {"llama-70b", "attn_out", 8192,          8192},
  {"llama-70b", "ffn_up",  28672,          8192},
  {"llama-70b", "ffn_down", 8192,         28672},
};

static const std::vector<int> M_VALUES = {1, 2, 4, 8, 16, 32};

static const int   WARMUP = 10;
static const int   ITERS  = 100;

int main(int argc, char** argv) {
  bool csv = false;
  for (int i = 1; i < argc; i++) {
    if (std::strcmp(argv[i], "--csv") == 0) csv = true;
    else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
      printf("Usage: %s [--csv]\n", argv[0]);
      return 0;
    }
  }

  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));
  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  // Pre-allocate the largest buffers we'll need so we don't churn allocations.
  // Largest GEMM: ffn_up of 70b at M=32 -> A:[32,8192] B:[8192,28672] C:[32,28672].
  size_t max_M = 0; for (int m : M_VALUES) max_M = std::max<size_t>(max_M, m);
  size_t max_N = 0, max_K = 0;
  for (const auto& s : SHAPES) {
    max_N = std::max<size_t>(max_N, (size_t)s.N);
    max_K = std::max<size_t>(max_K, (size_t)s.K);
  }
  half *dA = nullptr, *dB = nullptr, *dC = nullptr;
  CHECK_CUDA(cudaMalloc(&dA, max_M * max_K * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&dB, max_K * max_N * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&dC, max_M * max_N * sizeof(half)));
  CHECK_CUDA(cudaMemset(dA, 0, max_M * max_K * sizeof(half)));
  CHECK_CUDA(cudaMemset(dB, 0, max_K * max_N * sizeof(half)));

  if (csv) {
    printf("model,kind,M,N,K,ms,gflops,achieved_bw_GBs,arith_intensity\n");
  } else {
    printf("%-10s %-10s %5s %6s %6s %10s %10s %10s %10s\n",
           "model","kind","M","N","K","ms","GFLOP/s","BW GB/s","FLOP/B");
    printf("%-10s %-10s %5s %6s %6s %10s %10s %10s %10s\n",
           "-----","-----","-","-","-","--","-------","-------","------");
  }

  const half alpha = __float2half(1.0f);
  const half beta  = __float2half(0.0f);

  for (const auto& s : SHAPES) {
    for (int M : M_VALUES) {
      // C[M,N] = A[M,K] * B[K,N], row-major.
      // Express with cuBLAS column-major as B^T * A^T -> C^T.
      // We just want timing, so ignore correctness and set up a valid call:
      //   m=N, n=M, k=K, A_arg=B(K,N), B_arg=A(M,K)
      const int m = s.N, n = M, k = s.K;

      // Warmup
      for (int i = 0; i < WARMUP; i++) {
        CHECK_CUBLAS(cublasGemmEx(
          handle, CUBLAS_OP_N, CUBLAS_OP_N,
          m, n, k,
          &alpha,
          dB, CUDA_R_16F, m,
          dA, CUDA_R_16F, k,
          &beta,
          dC, CUDA_R_16F, m,
          CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      }
      CHECK_CUDA(cudaDeviceSynchronize());

      // Timed
      CHECK_CUDA(cudaEventRecord(start));
      for (int i = 0; i < ITERS; i++) {
        CHECK_CUBLAS(cublasGemmEx(
          handle, CUBLAS_OP_N, CUBLAS_OP_N,
          m, n, k,
          &alpha,
          dB, CUDA_R_16F, m,
          dA, CUDA_R_16F, k,
          &beta,
          dC, CUDA_R_16F, m,
          CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      }
      CHECK_CUDA(cudaEventRecord(stop));
      CHECK_CUDA(cudaEventSynchronize(stop));

      float total_ms = 0.f;
      CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
      double ms_per = total_ms / ITERS;

      double flops  = 2.0 * (double)M * s.N * s.K;
      double bytes  = ((double)M * s.K + (double)s.K * s.N + (double)M * s.N) * sizeof(half);
      double gflops = flops / (ms_per * 1e-3) / 1e9;
      double gbs    = bytes / (ms_per * 1e-3) / 1e9;
      double ai     = flops / bytes;

      if (csv) {
        printf("%s,%s,%d,%d,%d,%.4f,%.2f,%.2f,%.2f\n",
               s.model, s.kind, M, s.N, s.K, ms_per, gflops, gbs, ai);
      } else {
        printf("%-10s %-10s %5d %6d %6d %10.4f %10.2f %10.2f %10.2f\n",
               s.model, s.kind, M, s.N, s.K, ms_per, gflops, gbs, ai);
      }
    }
    if (!csv) printf("\n");
  }

  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  cudaEventDestroy(start); cudaEventDestroy(stop);
  cublasDestroy(handle);
  return 0;
}
