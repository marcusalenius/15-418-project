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
#include "bench_common.h"

#ifdef USE_NCCL
#include <thread>
#include <nccl.h>
#include "forward_tp.h"
#endif


// ---------------------------------------------------------------------------
// Single-GPU
// ---------------------------------------------------------------------------

// Time `bench_iters` AR-decode runs of N_tokens forward passes (each at M=1).
// Returns ms per outer iter (i.e. per N-token decode).
static float bench_ar_decode_single(
  const ModelConfig& cfg,
  int N_tokens,
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

  ActivationBuffers buf = alloc_activations(cfg, 1);

  for (int i = 0; i < warmup_iters; i++)
    forward_model(handle, cfg, all_weights.data(), 1,
                  buf.x, buf.scratch1, buf.scratch2);
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < bench_iters; i++)
    for (int t = 0; t < N_tokens; t++)
      forward_model(handle, cfg, all_weights.data(), 1,
                    buf.x, buf.scratch1, buf.scratch2);
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

// Per-layer fallback for models that don't fit on one GPU. Returns ms per
// single forward pass at M=1 (extrapolated to n_layers).
static float bench_step_layer_single(
  const ModelConfig& cfg,
  int warmup_iters,
  int bench_iters
) {
  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));
  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  LayerWeights weights;
  alloc_layer_weights(weights, cfg);
  ActivationBuffers buf = alloc_activations(cfg, 1);

  for (int i = 0; i < warmup_iters; i++)
    forward_layer(handle, cfg, weights, 1,
                  buf.x, buf.scratch1, buf.scratch2);
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < bench_iters; i++)
    forward_layer(handle, cfg, weights, 1,
                  buf.x, buf.scratch1, buf.scratch2);
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float total_ms = 0;
  CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  free_activations(buf);
  free_layer_weights(weights);
  CHECK_CUBLAS(cublasDestroy(handle));

  return (total_ms / bench_iters) * cfg.n_layers;
}


// ---------------------------------------------------------------------------
// TP
// ---------------------------------------------------------------------------

#ifdef USE_NCCL

static float bench_ar_decode_tp(
  const ModelConfig& cfg,
  int T,
  int N_tokens,
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

      int L = cfg.n_layers;
      std::vector<TPLayerWeights> all_weights(L);
      for (int l = 0; l < L; l++)
        alloc_tp_layer_weights(all_weights[l], cfg, T);

      ActivationBuffers buf = alloc_activations_tp(cfg, 1, T);

      for (int i = 0; i < warmup_iters; i++)
        forward_model_tp(ctx.handle, ctx.comm, ctx.stream,
                         cfg, all_weights.data(), T, 1,
                         buf.x, buf.scratch1, buf.scratch2);
      CHECK_CUDA(cudaStreamSynchronize(ctx.stream));

      cudaEvent_t start, stop;
      CHECK_CUDA(cudaEventCreate(&start));
      CHECK_CUDA(cudaEventCreate(&stop));

      CHECK_CUDA(cudaEventRecord(start, ctx.stream));
      for (int i = 0; i < bench_iters; i++)
        for (int t = 0; t < N_tokens; t++)
          forward_model_tp(ctx.handle, ctx.comm, ctx.stream,
                           cfg, all_weights.data(), T, 1,
                           buf.x, buf.scratch1, buf.scratch2);
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

  for (auto& t : threads) t.join();
  return results[0];
}

static float bench_step_layer_tp(
  const ModelConfig& cfg,
  int T,
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
      ActivationBuffers buf = alloc_activations_tp(cfg, 1, T);

      for (int i = 0; i < warmup_iters; i++)
        forward_layer_tp(ctx.handle, ctx.comm, ctx.stream,
                         cfg, weights, T, 1,
                         buf.x, buf.scratch1, buf.scratch2);
      CHECK_CUDA(cudaStreamSynchronize(ctx.stream));

      cudaEvent_t start, stop;
      CHECK_CUDA(cudaEventCreate(&start));
      CHECK_CUDA(cudaEventCreate(&stop));

      CHECK_CUDA(cudaEventRecord(start, ctx.stream));
      for (int i = 0; i < bench_iters; i++)
        forward_layer_tp(ctx.handle, ctx.comm, ctx.stream,
                         cfg, weights, T, 1,
                         buf.x, buf.scratch1, buf.scratch2);
      CHECK_CUDA(cudaEventRecord(stop, ctx.stream));
      CHECK_CUDA(cudaEventSynchronize(stop));

      float total_ms = 0;
      CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
      results[r] = (total_ms / bench_iters) * cfg.n_layers;

      CHECK_CUDA(cudaEventDestroy(start));
      CHECK_CUDA(cudaEventDestroy(stop));
      free_activations(buf);
      free_tp_layer_weights(weights);
      destroy_gpu(ctx);
    });
  }

  for (auto& t : threads) t.join();
  return results[0];
}

#endif // USE_NCCL


// ---------------------------------------------------------------------------
// Dispatch helpers
// ---------------------------------------------------------------------------

// One forward pass at M=1: ms.
static float measure_step_ms(
  const ModelConfig& cfg,
  int tp,
  int warmup_iters,
  int bench_iters,
  bool fits
) {
  if (fits) {
    if (tp == 1)
      return bench_ar_decode_single(cfg, 1, warmup_iters, bench_iters);
    #ifdef USE_NCCL
      return bench_ar_decode_tp(cfg, tp, 1, warmup_iters, bench_iters);
    #else
      fprintf(stderr, "Error: TP>1 requires building with USE_NCCL=1\n");
      exit(1);
    #endif
  } else {
    if (tp == 1)
      return bench_step_layer_single(cfg, warmup_iters, bench_iters * 5);
    #ifdef USE_NCCL
      return bench_step_layer_tp(cfg, tp, warmup_iters, bench_iters * 5);
    #else
      fprintf(stderr, "Error: TP>1 requires building with USE_NCCL=1\n");
      exit(1);
    #endif
  }
}

// Full N-token decode
static float measure_e2e_ms(
  const ModelConfig& cfg,
  int tp,
  int N_tokens,
  int warmup_iters,
  int bench_iters,
  bool fits
) {
  if (fits) {
    if (tp == 1)
      return bench_ar_decode_single(cfg, N_tokens, warmup_iters, bench_iters);
    #ifdef USE_NCCL
      return bench_ar_decode_tp(cfg, tp, N_tokens, warmup_iters, bench_iters);
    #else
      fprintf(stderr, "Error: TP>1 requires building with USE_NCCL=1\n");
      exit(1);
    #endif
  } else {
    // step_ms is one full forward pass; extrapolate to N tokens.
    float step_ms = measure_step_ms(cfg, tp, warmup_iters, bench_iters, fits);
    return step_ms * N_tokens;
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
  int bench_iters,
  const BenchOpts& opts
) {
  bool fits = model_fits(cfg, tp);

  BenchInfo info;
  info.mode = "ar";
  info.target = &cfg;
  info.tp = tp;
  info.N = N_tokens;
  info.warmup = warmup_iters;
  info.iters = bench_iters;
  info.target_fits = fits;

  print_bench_header(info);

  ComponentTimes ct;
  if (!opts.skip_component) {
    ct.target_step_ms = measure_step_ms(cfg, tp, warmup_iters, bench_iters, fits);
  }
  print_components(info, ct);

  BenchResult br;
  if (!opts.skip_e2e) {
    float e2e_ms = measure_e2e_ms(cfg, tp, N_tokens,
                                  warmup_iters, bench_iters, fits);
    br.e2e_ms          = e2e_ms;
    br.avg_tokens      = static_cast<float>(N_tokens);
    br.avg_rounds      = static_cast<float>(N_tokens);  // 1 token per round
    br.throughput_tok_s = e2e_ms > 0 ? N_tokens * 1000.0f / e2e_ms : 0.0f;
  }
  print_result(info, br);

  if (opts.csv) print_csv_row(info, ct, br);
}
