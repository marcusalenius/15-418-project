// bench_sd.cu
// SD benchmark implementations.

#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include "check.h"
#include "config.h"
#include "gpu_context.h"
#include "forward_single.h"

#ifdef USE_NCCL
#include <thread>
#include <barrier>
#include <atomic>
#include <nccl.h>
#include "forward_tp.h"
#endif


// ---------------------------------------------------------------------------
// Single-GPU component benchmarks
// ---------------------------------------------------------------------------

static float bench_draft_single(
  const ModelConfig& draft_cfg,
  int K,
  int warmup_iters,
  int bench_iters,
  bool fits
) {
  cudaSetDevice(0);
  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));
  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  if (fits) {
    int L = draft_cfg.n_layers;
    std::vector<LayerWeights> weights(L);
    for (int l = 0; l < L; l++)
      alloc_layer_weights(weights[l], draft_cfg);
    ActivationBuffers buf = alloc_activations(draft_cfg, 1);

    for (int i = 0; i < warmup_iters; i++)
      for (int k = 0; k < K; k++)
        forward_model(handle, draft_cfg, weights.data(), 1,
                      buf.x, buf.scratch1, buf.scratch2);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < bench_iters; i++)
      for (int k = 0; k < K; k++)
        forward_model(handle, draft_cfg, weights.data(), 1,
                      buf.x, buf.scratch1, buf.scratch2);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    free_activations(buf);
    for (int l = 0; l < L; l++)
      free_layer_weights(weights[l]);
    CHECK_CUBLAS(cublasDestroy(handle));
    return total_ms / bench_iters;
  } else {
    LayerWeights weights;
    alloc_layer_weights(weights, draft_cfg);
    ActivationBuffers buf = alloc_activations(draft_cfg, 1);

    for (int i = 0; i < warmup_iters; i++)
      forward_layer(handle, draft_cfg, weights, 1,
                    buf.x, buf.scratch1, buf.scratch2);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int layer_iters = bench_iters * 5;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < layer_iters; i++)
      forward_layer(handle, draft_cfg, weights, 1,
                    buf.x, buf.scratch1, buf.scratch2);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
    float layer_ms = total_ms / layer_iters;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    free_activations(buf);
    free_layer_weights(weights);
    CHECK_CUBLAS(cublasDestroy(handle));
    return layer_ms * draft_cfg.n_layers * K;
  }
}

static float bench_target_single(
  const ModelConfig& target_cfg,
  int M,
  int warmup_iters,
  int bench_iters,
  bool fits
) {
  cudaSetDevice(0);
  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));
  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  if (fits) {
    int L = target_cfg.n_layers;
    std::vector<LayerWeights> weights(L);
    for (int l = 0; l < L; l++)
      alloc_layer_weights(weights[l], target_cfg);
    ActivationBuffers buf = alloc_activations(target_cfg, M);

    for (int i = 0; i < warmup_iters; i++)
      forward_model(handle, target_cfg, weights.data(), M,
                    buf.x, buf.scratch1, buf.scratch2);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < bench_iters; i++)
      forward_model(handle, target_cfg, weights.data(), M,
                    buf.x, buf.scratch1, buf.scratch2);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    free_activations(buf);
    for (int l = 0; l < L; l++)
      free_layer_weights(weights[l]);
    CHECK_CUBLAS(cublasDestroy(handle));
    return total_ms / bench_iters;
  } else {
    LayerWeights weights;
    alloc_layer_weights(weights, target_cfg);
    ActivationBuffers buf = alloc_activations(target_cfg, M);

    for (int i = 0; i < warmup_iters; i++)
      forward_layer(handle, target_cfg, weights, M,
                    buf.x, buf.scratch1, buf.scratch2);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int layer_iters = bench_iters * 5;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < layer_iters; i++)
      forward_layer(handle, target_cfg, weights, M,
                    buf.x, buf.scratch1, buf.scratch2);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
    float layer_ms = total_ms / layer_iters;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    free_activations(buf);
    free_layer_weights(weights);
    CHECK_CUBLAS(cublasDestroy(handle));
    return layer_ms * target_cfg.n_layers;
  }
}

static float bench_sd_e2e_single(
  const ModelConfig& target_cfg,
  const ModelConfig& draft_cfg,
  int K,
  int N,
  float alpha,
  int warmup_iters,
  int bench_iters
) {
  cudaSetDevice(0);
  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));
  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  int tL = target_cfg.n_layers;
  int dL = draft_cfg.n_layers;

  std::vector<LayerWeights> target_w(tL);
  for (int l = 0; l < tL; l++)
    alloc_layer_weights(target_w[l], target_cfg);
  ActivationBuffers target_buf = alloc_activations(target_cfg, K);

  std::vector<LayerWeights> draft_w(dL);
  for (int l = 0; l < dL; l++)
    alloc_layer_weights(draft_w[l], draft_cfg);
  ActivationBuffers draft_buf = alloc_activations(draft_cfg, 1);

  std::mt19937 rng(42);
  std::bernoulli_distribution accept_dist(alpha);

  for (int i = 0; i < warmup_iters; i++) {
    for (int k = 0; k < K; k++)
      forward_model(handle, draft_cfg, draft_w.data(), 1,
                    draft_buf.x, draft_buf.scratch1, draft_buf.scratch2);
    forward_model(handle, target_cfg, target_w.data(), K,
                  target_buf.x, target_buf.scratch1, target_buf.scratch2);
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  int total_tokens = 0;
  int total_rounds = 0;

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < bench_iters; i++) {
    int tokens = 0;
    while (tokens < N) {
      for (int k = 0; k < K; k++)
        forward_model(handle, draft_cfg, draft_w.data(), 1,
                      draft_buf.x, draft_buf.scratch1, draft_buf.scratch2);
      forward_model(handle, target_cfg, target_w.data(), K,
                    target_buf.x, target_buf.scratch1, target_buf.scratch2);

      int accepted = 0;
      for (int k = 0; k < K; k++) {
        if (accept_dist(rng))
          accepted++;
        else
          break;
      }
      tokens += accepted + 1;
      total_rounds++;
    }
    total_tokens += tokens;
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float total_ms = 0;
  CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  free_activations(draft_buf);
  for (int l = 0; l < dL; l++)
    free_layer_weights(draft_w[l]);
  free_activations(target_buf);
  for (int l = 0; l < tL; l++)
    free_layer_weights(target_w[l]);
  CHECK_CUBLAS(cublasDestroy(handle));

  float avg_ms = total_ms / bench_iters;
  float avg_tokens = static_cast<float>(total_tokens) / bench_iters;
  float avg_rounds = static_cast<float>(total_rounds) / bench_iters;

  printf("    Avg tokens generated: %.1f (target %d)\n", avg_tokens, N);
  printf("    Avg rounds:           %.1f\n", avg_rounds);
  printf("    Avg tokens/round:     %.2f\n", avg_tokens / avg_rounds);
  printf("    Total time:           %.2f ms\n", avg_ms);
  printf("    Throughput:           %.1f tok/s\n", avg_tokens * 1000.0f / avg_ms);

  return avg_ms;
}


// ---------------------------------------------------------------------------
// TP component benchmarks
// ---------------------------------------------------------------------------

#ifdef USE_NCCL

static float bench_target_tp(
  const ModelConfig& target_cfg,
  int T,
  int M,
  int warmup_iters,
  int bench_iters,
  bool fits
) {
  ncclUniqueId id;
  CHECK_NCCL(ncclGetUniqueId(&id));

  std::vector<float> results(T);
  std::vector<std::thread> threads;

  if (fits) {
    for (int r = 0; r < T; r++) {
      threads.emplace_back([&, r]() {
        GPUContext ctx = init_gpu(r, T, id);

        int L = target_cfg.n_layers;
        std::vector<TPLayerWeights> weights(L);
        for (int l = 0; l < L; l++)
          alloc_tp_layer_weights(weights[l], target_cfg, T);
        ActivationBuffers buf = alloc_activations_tp(target_cfg, M, T);

        for (int i = 0; i < warmup_iters; i++)
          forward_model_tp(ctx.handle, ctx.comm, ctx.stream,
                           target_cfg, weights.data(), T, M,
                           buf.x, buf.scratch1, buf.scratch2);
        CHECK_CUDA(cudaStreamSynchronize(ctx.stream));

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start, ctx.stream));
        for (int i = 0; i < bench_iters; i++)
          forward_model_tp(ctx.handle, ctx.comm, ctx.stream,
                           target_cfg, weights.data(), T, M,
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
          free_tp_layer_weights(weights[l]);
        destroy_gpu(ctx);
      });
    }
  } else {
    for (int r = 0; r < T; r++) {
      threads.emplace_back([&, r]() {
        GPUContext ctx = init_gpu(r, T, id);

        TPLayerWeights weights;
        alloc_tp_layer_weights(weights, target_cfg, T);
        ActivationBuffers buf = alloc_activations_tp(target_cfg, M, T);

        for (int i = 0; i < warmup_iters; i++)
          forward_layer_tp(ctx.handle, ctx.comm, ctx.stream,
                           target_cfg, weights, T, M,
                           buf.x, buf.scratch1, buf.scratch2);
        CHECK_CUDA(cudaStreamSynchronize(ctx.stream));

        int layer_iters = bench_iters * 5;
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start, ctx.stream));
        for (int i = 0; i < layer_iters; i++)
          forward_layer_tp(ctx.handle, ctx.comm, ctx.stream,
                           target_cfg, weights, T, M,
                           buf.x, buf.scratch1, buf.scratch2);
        CHECK_CUDA(cudaEventRecord(stop, ctx.stream));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float total_ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
        results[r] = (total_ms / layer_iters) * target_cfg.n_layers;

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        free_activations(buf);
        free_tp_layer_weights(weights);
        destroy_gpu(ctx);
      });
    }
  }

  for (auto& t : threads) t.join();
  return results[0];
}

// Shard the target model across all T GPUS.
// The draft model runs only on rank 0.
static float bench_sd_e2e_tp(
  const ModelConfig& target_cfg,
  const ModelConfig& draft_cfg,
  int T,
  int K,
  int N,
  float alpha,
  int warmup_iters,
  int bench_iters
) {
  ncclUniqueId id;
  CHECK_NCCL(ncclGetUniqueId(&id));

  std::vector<float> results(T);
  std::vector<std::thread> threads;

  std::barrier sync(T);
  std::atomic<bool> round_done{false};

  // Only rank 0 tracks these
  int total_tokens = 0;
  int total_rounds = 0;

  for (int r = 0; r < T; r++) {
    threads.emplace_back([&, r]() {
      GPUContext ctx = init_gpu(r, T, id);

      // Target weights (all ranks, TP-sharded)
      int tL = target_cfg.n_layers;
      std::vector<TPLayerWeights> target_w(tL);
      for (int l = 0; l < tL; l++)
        alloc_tp_layer_weights(target_w[l], target_cfg, T);
      ActivationBuffers target_buf = alloc_activations_tp(target_cfg, K, T);

      // Draft weights (rank 0 only, single-GPU)
      std::vector<LayerWeights> draft_w;
      ActivationBuffers draft_buf = {nullptr, nullptr, nullptr};
      cublasHandle_t draft_handle = nullptr;
      cudaStream_t draft_stream = nullptr;
      if (r == 0) {
        int dL = draft_cfg.n_layers;
        draft_w.resize(dL);
        for (int l = 0; l < dL; l++)
          alloc_layer_weights(draft_w[l], draft_cfg);
        draft_buf = alloc_activations(draft_cfg, 1);

        // Separate handle and stream for draft
        CHECK_CUBLAS(cublasCreate(&draft_handle));
        CHECK_CUBLAS(cublasSetMathMode(draft_handle, CUBLAS_TENSOR_OP_MATH));
        CHECK_CUDA(cudaStreamCreate(&draft_stream));
        CHECK_CUBLAS(cublasSetStream(draft_handle, draft_stream));
      }

      std::mt19937 rng(42);
      std::bernoulli_distribution accept_dist(alpha);

      // Warmup
      for (int i = 0; i < warmup_iters; i++) {
        if (r == 0) {
          for (int k = 0; k < K; k++)
            forward_model(draft_handle, draft_cfg, draft_w.data(), 1,
                          draft_buf.x, draft_buf.scratch1, draft_buf.scratch2);
          CHECK_CUDA(cudaStreamSynchronize(draft_stream));
        }
        sync.arrive_and_wait();
        forward_model_tp(ctx.handle, ctx.comm, ctx.stream,
                         target_cfg, target_w.data(), T, K,
                         target_buf.x, target_buf.scratch1, target_buf.scratch2);
        CHECK_CUDA(cudaStreamSynchronize(ctx.stream));
        sync.arrive_and_wait();
      }

      cudaEvent_t start, stop;
      CHECK_CUDA(cudaEventCreate(&start));
      CHECK_CUDA(cudaEventCreate(&stop));

      CHECK_CUDA(cudaEventRecord(start, ctx.stream));
      for (int i = 0; i < bench_iters; i++) {
        int local_tokens = 0;
        if (r == 0) round_done.store(false);
        sync.arrive_and_wait();

        while (true) {
          if (r == 0) {
            for (int k = 0; k < K; k++)
              forward_model(draft_handle, draft_cfg, draft_w.data(), 1,
                            draft_buf.x, draft_buf.scratch1, draft_buf.scratch2);
            CHECK_CUDA(cudaStreamSynchronize(draft_stream));
          }
          sync.arrive_and_wait();

          forward_model_tp(ctx.handle, ctx.comm, ctx.stream,
                           target_cfg, target_w.data(), T, K,
                           target_buf.x, target_buf.scratch1, target_buf.scratch2);
          CHECK_CUDA(cudaStreamSynchronize(ctx.stream));
          sync.arrive_and_wait();

          if (r == 0) {
            int accepted = 0;
            for (int k = 0; k < K; k++) {
              if (accept_dist(rng))
                accepted++;
              else
                break;
            }
            local_tokens += accepted + 1;
            total_rounds++;

            if (local_tokens >= N) {
              total_tokens += local_tokens;
              round_done.store(true);
            }
          }
          sync.arrive_and_wait();

          if (round_done.load()) break;
        }
      }
      CHECK_CUDA(cudaEventRecord(stop, ctx.stream));
      CHECK_CUDA(cudaEventSynchronize(stop));

      float total_ms = 0;
      CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
      results[r] = total_ms;

      CHECK_CUDA(cudaEventDestroy(start));
      CHECK_CUDA(cudaEventDestroy(stop));
      free_activations(target_buf);
      for (int l = 0; l < tL; l++)
        free_tp_layer_weights(target_w[l]);
      if (r == 0) {
        free_activations(draft_buf);
        for (int l = 0; l < (int)draft_w.size(); l++)
          free_layer_weights(draft_w[l]);
        CHECK_CUBLAS(cublasDestroy(draft_handle));
        CHECK_CUDA(cudaStreamDestroy(draft_stream));
      }
      destroy_gpu(ctx);
    });
  }

  for (auto& t : threads) t.join();

  float avg_ms = results[0] / bench_iters;
  float avg_tokens = static_cast<float>(total_tokens) / bench_iters;
  float avg_rounds = static_cast<float>(total_rounds) / bench_iters;

  printf("    Avg tokens generated: %.1f (target %d)\n", avg_tokens, N);
  printf("    Avg rounds:           %.1f\n", avg_rounds);
  printf("    Avg tokens/round:     %.2f\n", avg_tokens / avg_rounds);
  printf("    Total time:           %.2f ms\n", avg_ms);
  printf("    Throughput:           %.1f tok/s\n", avg_tokens * 1000.0f / avg_ms);

  return avg_ms;
}

#endif // USE_NCCL


// ---------------------------------------------------------------------------
// Analytical helpers
// ---------------------------------------------------------------------------

static double expected_tokens(int K, double alpha) {
  if (alpha < 1e-9) return 1.0;
  if (std::abs(alpha - 1.0) < 1e-9) return K + 1.0;
  return (1.0 - std::pow(alpha, K + 1)) / (1.0 - alpha);
}


// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void run_sd_benchmark(
  const ModelConfig& target_cfg,
  const ModelConfig& draft_cfg,
  int tp,
  int K,
  int N,
  float alpha,
  int warmup_iters,
  int bench_iters
) {
  bool target_fits = model_fits(target_cfg, tp);
  bool draft_fits = model_fits(draft_cfg, 1);

  printf("--- SD Benchmark ---\n");
  printf("  Target: %s (d=%d, L=%d)\n",
         target_cfg.name.c_str(), target_cfg.d_model, target_cfg.n_layers);
  printf("  Draft:  %s (d=%d, L=%d)\n",
         draft_cfg.name.c_str(), draft_cfg.d_model, draft_cfg.n_layers);
  printf("  TP=%d, K=%d, alpha=%.2f, N=%d\n\n", tp, K, alpha, N);

  size_t target_bytes = (tp == 1)
    ? target_cfg.total_weight_bytes()
    : target_cfg.total_tp_weight_bytes(tp);
  size_t draft_bytes = draft_cfg.total_weight_bytes();
  printf("  Target weight footprint%s: %.2f GB | Fits: %s\n",
         tp > 1 ? " (per GPU)" : "",
         target_bytes / (1024.0 * 1024 * 1024),
         target_fits ? "YES" : "NO (layer estimate)");
  printf("  Draft weight footprint:       %.2f GB | Fits: %s\n\n",
         draft_bytes / (1024.0 * 1024 * 1024),
         draft_fits ? "YES" : "NO (layer estimate)");

  // -----------------------------------------------------------------------
  // Component timing
  // -----------------------------------------------------------------------
  printf("  [Component Timing]\n");

  float draft_total_ms;
  if (tp == 1) {
    draft_total_ms = bench_draft_single(draft_cfg, K, warmup_iters, bench_iters, draft_fits);
  } else {
    draft_total_ms = bench_draft_single(draft_cfg, K, warmup_iters, bench_iters, draft_fits);
  }
  float draft_step_ms = draft_total_ms / K;

  float target_ar_ms, target_verify_ms;
  if (tp == 1) {
    target_ar_ms     = bench_target_single(target_cfg, 1, warmup_iters, bench_iters, target_fits);
    target_verify_ms = bench_target_single(target_cfg, K, warmup_iters, bench_iters, target_fits);
  } else {
    #ifdef USE_NCCL
      target_ar_ms     = bench_target_tp(target_cfg, tp, 1, warmup_iters, bench_iters, target_fits);
      target_verify_ms = bench_target_tp(target_cfg, tp, K, warmup_iters, bench_iters, target_fits);
    #else
      fprintf(stderr, "Error: TP>1 requires building with USE_NCCL=1\n");
      exit(1);
    #endif
  }

  float sd_round_ms = draft_total_ms + target_verify_ms;
  float t_sd_ratio  = draft_total_ms / target_verify_ms;

  printf("    Draft step (M=1):       %.3f ms\n", draft_step_ms);
  printf("    Draft total (K=%d):      %.3f ms\n", K, draft_total_ms);
  printf("    Target AR step (M=1):   %.3f ms\n", target_ar_ms);
  printf("    Target verify (M=%d):    %.3f ms\n", K, target_verify_ms);
  printf("    SD round time:          %.3f ms\n", sd_round_ms);
  printf("    T_SD ratio:             %.4f\n\n", t_sd_ratio);

  // -----------------------------------------------------------------------
  // Verification window characterization
  // -----------------------------------------------------------------------
  printf("  [Verification Window]\n");
  int max_draft_passes = static_cast<int>(target_verify_ms / draft_step_ms);
  printf("    Target verify = %.3f ms, draft step = %.3f ms\n",
         target_verify_ms, draft_step_ms);
  printf("    Max draft passes in window: %d\n", max_draft_passes);
  printf("    Max fan-out budget (F):     %d  (at K=%d)\n\n",
         max_draft_passes / K, K);

  // -----------------------------------------------------------------------
  // Analytical predictions
  // -----------------------------------------------------------------------
  printf("  [Analytical Predictions]\n");
  printf("    %5s  %14s  %12s  %12s  %9s\n",
         "alpha", "E[tok/round]", "SD tok/s", "AR tok/s", "Speedup");
  printf("    %5s  %14s  %12s  %12s  %9s\n",
         "-----", "--------------", "------------", "------------", "---------");

  float ar_toks_per_sec = 1000.0f / target_ar_ms;
  for (float a : {0.5f, 0.7f, 0.9f}) {
    double e_tok = expected_tokens(K, a);
    float sd_toks_per_sec = static_cast<float>(e_tok) * 1000.0f / sd_round_ms;
    float speedup = sd_toks_per_sec / ar_toks_per_sec;
    printf("    %5.2f  %14.2f  %10.1f    %10.1f    %7.2fx\n",
           a, e_tok, sd_toks_per_sec, ar_toks_per_sec, speedup);
  }
  printf("\n");

  // -----------------------------------------------------------------------
  // End-to-end simulation
  // -----------------------------------------------------------------------
  bool both_fit = target_fits && draft_fits;

  if (tp == 1 && both_fit) {
    printf("  [End-to-End Simulation] (alpha=%.2f, N=%d)\n", alpha, N);
    float e2e_ms = bench_sd_e2e_single(
      target_cfg, draft_cfg, K, N, alpha, warmup_iters, bench_iters);

    float predicted_e = static_cast<float>(expected_tokens(K, alpha));
    float predicted_throughput = predicted_e * 1000.0f / sd_round_ms;
    printf("    Predicted throughput:  %.1f tok/s\n\n", predicted_throughput);

    (void)e2e_ms;
  } else if (tp > 1) {
    #ifdef USE_NCCL
      printf("  [End-to-End Simulation TP] (alpha=%.2f, N=%d)\n", alpha, N);
      float e2e_ms = bench_sd_e2e_tp(
        target_cfg, draft_cfg, tp, K, N, alpha, warmup_iters, bench_iters);

      float predicted_e = static_cast<float>(expected_tokens(K, alpha));
      float predicted_throughput = predicted_e * 1000.0f / sd_round_ms;
      printf("    Predicted throughput:  %.1f tok/s\n\n", predicted_throughput);

      (void)e2e_ms;
    #endif
  } else {
    printf("  [End-to-End Simulation] skipped (model does not fit in memory)\n\n");
  }

  // -----------------------------------------------------------------------
  // SD speedup summary
  // -----------------------------------------------------------------------
  double e_tok = expected_tokens(K, alpha);
  float sd_toks_per_sec = static_cast<float>(e_tok) * 1000.0f / sd_round_ms;
  float speedup = sd_toks_per_sec / ar_toks_per_sec;

  printf("  [Summary] (alpha=%.2f, K=%d)\n", alpha, K);
  printf("    E[tokens/round]: %.2f\n", e_tok);
  printf("    AR throughput:   %.1f tok/s (%.3f ms/tok)\n",
         ar_toks_per_sec, target_ar_ms);
  printf("    SD throughput:   %.1f tok/s (%.3f ms/round)\n",
         sd_toks_per_sec, sd_round_ms);
  printf("    SD speedup:      %.2fx over AR\n\n", speedup);
}
