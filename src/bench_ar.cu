// bench_ar.cu
// AR decode benchmark implementations.

#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include "check.h"
#include "config.h"
#include "gpu_context.h"
#include "forward_single.h"

#ifdef USE_NCCL
#include <thread>
#include <nccl.h>
#include "forward_tp.h"
#endif


// ---------------------------------------------------------------------------
// Single-GPU benchmarks
// ---------------------------------------------------------------------------

static float bench_ar_decode_single(
  const ModelConfig& cfg,
  int N_tokens,
  int M,
  int warmup_iters,
  int bench_iters
) {
  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));
  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  int L = cfg.n_layers;
  std::vector<LayerWeights> all_weights(L);
  for (int l = 0; l < L; l++)
    alloc_layer_weights(all_weights[l], cfg);

  ActivationBuffers buf = alloc_activations(cfg, M);

  // Warmup
  for (int i = 0; i < warmup_iters; i++)
    forward_model(
      handle, cfg, all_weights.data(), M, buf.x, buf.scratch1, buf.scratch2
    );
  CHECK_CUDA(cudaDeviceSynchronize());

  // Timed
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < bench_iters; i++)
    for (int t = 0; t < N_tokens; t++)
      forward_model(
        handle, cfg, all_weights.data(), M, buf.x, buf.scratch1, buf.scratch2
      );
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float total_ms = 0;
  CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  free_activations(buf);
  for (int l = 0; l < L; l++)
    free_layer_weights(all_weights[l]);
  CHECK_CUBLAS(cublasDestroy(handle));

  return total_ms / bench_iters;
}

static float bench_layer_single(
  const ModelConfig& cfg,
  int M,
  int warmup_iters,
  int bench_iters
) {
  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));
  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  LayerWeights weights;
  alloc_layer_weights(weights, cfg);
  ActivationBuffers buf = alloc_activations(cfg, M);

  // Warmup
  for (int i = 0; i < warmup_iters; i++)
    forward_layer(
      handle, cfg, weights, M, buf.x, buf.scratch1, buf.scratch2
    );
  CHECK_CUDA(cudaDeviceSynchronize());

  // Timed
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < bench_iters; i++)
    forward_layer(
      handle, cfg, weights, M, buf.x, buf.scratch1, buf.scratch2
    );
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float total_ms = 0;
  CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  free_activations(buf);
  free_layer_weights(weights);
  CHECK_CUBLAS(cublasDestroy(handle));

  return total_ms / bench_iters;
}

// ---------------------------------------------------------------------------
// TP benchmarks
// ---------------------------------------------------------------------------

#ifdef USE_NCCL

static float bench_ar_decode_tp(
  const ModelConfig& cfg,
  int T,
  int N_tokens,
  int M,
  int warmup_iters,
  int bench_iters
) {
  ncclUniqueId id;
  CHECK_NCCL(ncclGetUniqueId(&id));

  std::vector<float> results(T);
  std::vector<std::thread> threads;

  for (int r = 0; r < T; r++) {
    // Each thread needs its own copy of r (0, 1, 2, ...)
    // but can share references to everything else since
    threads.emplace_back([&, r]() {
      GPUContext ctx = init_gpu(r, T, id);

      int L = cfg.n_layers;
      std::vector<TPLayerWeights> all_weights(L);
      for (int l = 0; l < L; l++)
        alloc_tp_layer_weights(all_weights[l], cfg, T);

      ActivationBuffers buf = alloc_activations_tp(cfg, M, T);

      // Warmup
      for (int i = 0; i < warmup_iters; i++)
        forward_model_tp(
          ctx.handle, ctx.comm, ctx.stream, cfg, all_weights.data(), T, M, 
          buf.x, buf.scratch1, buf.scratch2
        );
      CHECK_CUDA(cudaStreamSynchronize(ctx.stream));

      // Timed
      cudaEvent_t start, stop;
      CHECK_CUDA(cudaEventCreate(&start));
      CHECK_CUDA(cudaEventCreate(&stop));

      CHECK_CUDA(cudaEventRecord(start, ctx.stream));
      for (int i = 0; i < bench_iters; i++)
        for (int t = 0; t < N_tokens; t++)
          forward_model_tp(
            ctx.handle, ctx.comm, ctx.stream, cfg, all_weights.data(), T, M, 
            buf.x, buf.scratch1, buf.scratch2
          );
      CHECK_CUDA(cudaEventRecord(stop, ctx.stream));
      CHECK_CUDA(cudaEventSynchronize(stop));

      float total_ms = 0;
      CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
      results[r] = total_ms / bench_iters;

      CHECK_CUDA(cudaEventDestroy(start));
      CHECK_CUDA(cudaEventDestroy(stop));
      free_activations(buf);
      for (int l = 0; l < L; l++)
        free_tp_layer_weights(all_weights[l]);
      destroy_gpu(ctx);
    });
  }

  for (auto& t : threads)
    t.join();
  return results[0];
}

static float bench_layer_tp(
  const ModelConfig& cfg,
  int T,
  int M,
  int warmup_iters,
  int bench_iters
) {
  ncclUniqueId id;
  CHECK_NCCL(ncclGetUniqueId(&id));

  std::vector<float> results(T);
  std::vector<std::thread> threads;

  for (int r = 0; r < T; r++) {
    threads.emplace_back([&, r]() {
      GPUContext ctx = init_gpu(r, T, id);

      TPLayerWeights weights;
      alloc_tp_layer_weights(weights, cfg, T);
      ActivationBuffers buf = alloc_activations_tp(cfg, M, T);

      // Warmup
      for (int i = 0; i < warmup_iters; i++)
        forward_layer_tp(
          ctx.handle, ctx.comm, ctx.stream, cfg, weights, T, M,
          buf.x, buf.scratch1, buf.scratch2
        );
      CHECK_CUDA(cudaStreamSynchronize(ctx.stream));

      // Timed
      cudaEvent_t start, stop;
      CHECK_CUDA(cudaEventCreate(&start));
      CHECK_CUDA(cudaEventCreate(&stop));

      CHECK_CUDA(cudaEventRecord(start, ctx.stream));
      for (int i = 0; i < bench_iters; i++)
        forward_layer_tp(
          ctx.handle, ctx.comm, ctx.stream, cfg, weights, T, M,
          buf.x, buf.scratch1, buf.scratch2
        );
      CHECK_CUDA(cudaEventRecord(stop, ctx.stream));
      CHECK_CUDA(cudaEventSynchronize(stop));

      float total_ms = 0;
      CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
      results[r] = total_ms / bench_iters;

      CHECK_CUDA(cudaEventDestroy(start));
      CHECK_CUDA(cudaEventDestroy(stop));
      free_activations(buf);
      free_tp_layer_weights(weights);
      destroy_gpu(ctx);
    });
  }

  for (auto& t : threads)
    t.join();
  return results[0];
}

#endif // USE_NCCL

// ---------------------------------------------------------------------------
// Unified dispatch
// ---------------------------------------------------------------------------
static float measure_ar_decode(
  const ModelConfig& cfg,
  int tp,
  int N_tokens,
  int M,
  int warmup_iters,
  int bench_iters,
  bool fits
) {
  if (fits) {
    if (tp == 1) {
      return bench_ar_decode_single(cfg, N_tokens, M, warmup_iters, bench_iters);
    } else {
      #ifdef USE_NCCL
        return bench_ar_decode_tp(cfg, tp, N_tokens, M, warmup_iters, bench_iters);
      #else
        fprintf(stderr, "Error: TP>1 requires building with USE_NCCL=1\n");
        exit(1);
      #endif
    }
  } else {
    float layer_ms;
    if (tp == 1) {
      layer_ms = bench_layer_single(cfg, M, warmup_iters, bench_iters * 5);
    } else {
      #ifdef USE_NCCL
        layer_ms = bench_layer_tp(cfg, tp, M, warmup_iters, bench_iters * 5);
      #else
        fprintf(stderr, "Error: TP>1 requires building with USE_NCCL=1\n");
        exit(1);
      #endif
    }
    return layer_ms * cfg.n_layers * N_tokens;
  }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
void run_ar_benchmark(
  const ModelConfig& cfg,
  int tp,
  int N_tokens,
  int warmup_iters,
  int bench_iters
) {
  bool fits = model_fits(cfg, tp);

  // Print model info
  printf("--- %s (TP=%d) ---\n", cfg.name.c_str(), tp);
  printf("  d_model=%d, n_heads=%d, n_kv_heads=%d, head_dim=%d, d_ff=%d, layers=%d\n",
         cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.d_ff, cfg.n_layers);

  size_t weight_bytes = (tp == 1)
    ? cfg.total_weight_bytes()
    : cfg.total_tp_weight_bytes(tp);
  printf("  Weight footprint%s: %.2f GB | Fits: %s\n\n",
         tp > 1 ? " (per GPU)" : "",
         weight_bytes / (1024.0 * 1024 * 1024),
         fits ? "YES" : "NO (layer*L estimate)");

  // --- AR decode: generate N tokens, M=1 ---
  printf("  [AR Decode] Generate %d tokens (M=1, %d layers per step)\n",
         N_tokens, cfg.n_layers);

  float gen_ms = measure_ar_decode(cfg, tp, N_tokens, 1,
                                   warmup_iters, bench_iters, fits);
  float per_tok = gen_ms / N_tokens;

  if (fits) {
    printf("    Total: %.2f ms | Per-token: %.3f ms | Throughput: %.1f tok/s\n",
           gen_ms, per_tok, N_tokens * 1000.0f / gen_ms);
  } else {
    printf("    Per-token: %.3f ms | Throughput: %.1f tok/s (est)\n",
           per_tok, 1000.0f / per_tok);
  }
  printf("    Arith intensity (M=1): %.2f\n\n", cfg.arith_intensity(1));

  // --- Verification pass: M tokens in one forward pass ---
  printf("  [Verification] One forward pass with M tokens batched\n");
  printf("    %5s  %14s  %14s  %12s\n",
         "M", "pass (ms)", "throughput", "arith intens");
  printf("    %5s  %14s  %14s  %12s\n",
         "-----", "--------------", "--------------", "------------");

  for (int M : {2, 4, 5, 8, 16}) {
    float pass_ms = measure_ar_decode(cfg, tp, 1, M,
                                      warmup_iters, bench_iters, fits);
    if (fits) {
      printf("    %5d  %12.3f ms  %10.1f tok/s  %10.2f\n",
             M, pass_ms, M * 1000.0f / pass_ms, cfg.arith_intensity(M));
    } else {
      printf("    %5d  %12.3f ms  %10.1f tok/s  %10.2f  (est)\n",
             M, pass_ms, M * 1000.0f / pass_ms, cfg.arith_intensity(M));
    }
  }
  printf("\n");
}
