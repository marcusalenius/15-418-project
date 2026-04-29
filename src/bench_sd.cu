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
#include "bench_common.h"

#ifdef USE_NCCL
#include <thread>
#include <pthread.h>
#include <atomic>
#include <nccl.h>
#include "forward_tp.h"
#endif


// ---------------------------------------------------------------------------
// Single-GPU
// ---------------------------------------------------------------------------

// Returns ms for K back-to-back draft forward passes at M=1
static float bench_draft_total_single(
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

// Returns ms for one target forward pass at M tokens
static float bench_target_pass_single(
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

// Single-GPU SD end-to-end. Returns BenchResult populated.
static BenchResult bench_sd_e2e_single(
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

  BenchResult br;
  br.e2e_ms           = total_ms / bench_iters;
  br.avg_tokens       = static_cast<float>(total_tokens) / bench_iters;
  br.avg_rounds       = static_cast<float>(total_rounds) / bench_iters;
  br.throughput_tok_s = br.e2e_ms > 0 ? br.avg_tokens * 1000.0f / br.e2e_ms : 0.0f;
  return br;
}


// ---------------------------------------------------------------------------
// TP
// ---------------------------------------------------------------------------

#ifdef USE_NCCL

// Returns ms for one target forward pass at M tokens (TP-sharded)
static float bench_target_pass_tp(
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

// TP SD end-to-end. Target shards across T GPUs, draft runs on rank 0.
static BenchResult bench_sd_e2e_tp(
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

  pthread_barrier_t sync;
  pthread_barrier_init(&sync, nullptr, T);
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
        pthread_barrier_wait(&sync);
        forward_model_tp(ctx.handle, ctx.comm, ctx.stream,
                         target_cfg, target_w.data(), T, K,
                         target_buf.x, target_buf.scratch1, target_buf.scratch2);
        CHECK_CUDA(cudaStreamSynchronize(ctx.stream));
        pthread_barrier_wait(&sync);
      }

      cudaEvent_t start, stop;
      CHECK_CUDA(cudaEventCreate(&start));
      CHECK_CUDA(cudaEventCreate(&stop));

      CHECK_CUDA(cudaEventRecord(start, ctx.stream));
      for (int i = 0; i < bench_iters; i++) {
        int local_tokens = 0;
        pthread_barrier_wait(&sync);
        if (r == 0) round_done.store(false);
        pthread_barrier_wait(&sync);

        while (true) {
          if (r == 0) {
            for (int k = 0; k < K; k++)
              forward_model(draft_handle, draft_cfg, draft_w.data(), 1,
                            draft_buf.x, draft_buf.scratch1, draft_buf.scratch2);
            CHECK_CUDA(cudaStreamSynchronize(draft_stream));
          }
          pthread_barrier_wait(&sync);

          forward_model_tp(ctx.handle, ctx.comm, ctx.stream,
                           target_cfg, target_w.data(), T, K,
                           target_buf.x, target_buf.scratch1, target_buf.scratch2);
          CHECK_CUDA(cudaStreamSynchronize(ctx.stream));
          pthread_barrier_wait(&sync);

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
          pthread_barrier_wait(&sync);

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
  pthread_barrier_destroy(&sync);

  BenchResult br;
  br.e2e_ms           = results[0] / bench_iters;
  br.avg_tokens       = static_cast<float>(total_tokens) / bench_iters;
  br.avg_rounds       = static_cast<float>(total_rounds) / bench_iters;
  br.throughput_tok_s = br.e2e_ms > 0 ? br.avg_tokens * 1000.0f / br.e2e_ms : 0.0f;
  return br;
}

#endif // USE_NCCL


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
  int bench_iters,
  const BenchOpts& opts
) {
  bool target_fits = model_fits(target_cfg, tp);
  bool draft_fits = model_fits(draft_cfg, 1);

  BenchInfo info;
  info.mode = "sd";
  info.target = &target_cfg;
  info.draft  = &draft_cfg;
  info.tp = tp;
  info.K = K;
  info.N = N;
  info.alpha = alpha;
  info.warmup = warmup_iters;
  info.iters = bench_iters;
  info.target_fits = target_fits;
  info.draft_fits  = draft_fits;

  print_bench_header(info);

  // Component timing
  ComponentTimes ct;
  if (!opts.skip_component) {
    ct.draft_total_ms = bench_draft_total_single(
      draft_cfg, K, warmup_iters, bench_iters, draft_fits
    );
    ct.draft_step_ms = ct.draft_total_ms / K;

    if (tp == 1) {
      ct.target_step_ms = bench_target_pass_single(
        target_cfg, 1, warmup_iters, bench_iters, target_fits
      );
      ct.target_verify_ms = bench_target_pass_single(
        target_cfg, K, warmup_iters, bench_iters, target_fits
      );
    } else {
      #ifdef USE_NCCL
        ct.target_step_ms = bench_target_pass_tp(
          target_cfg, tp, 1, warmup_iters, bench_iters, target_fits
        );
        ct.target_verify_ms = bench_target_pass_tp(
          target_cfg, tp, K, warmup_iters, bench_iters, target_fits
        );
      #else
        fprintf(stderr, "Error: TP>1 requires building with USE_NCCL=1\n");
        exit(1);
      #endif
    }
  }
  print_components(info, ct);

  // End-to-end
  BenchResult br;
  if (!opts.skip_e2e) {
    bool both_fit = target_fits && draft_fits;
    if (!both_fit) {
      printf("  [End-to-End] skipped (model does not fit in memory)\n\n");
    } else if (tp == 1) {
      br = bench_sd_e2e_single(
        target_cfg, draft_cfg, K, N, alpha, warmup_iters, bench_iters
      );
    } else {
      #ifdef USE_NCCL
        br = bench_sd_e2e_tp(
          target_cfg, draft_cfg, tp, K, N, alpha, warmup_iters, bench_iters
        );
      #else
        fprintf(stderr, "Error: TP>1 requires building with USE_NCCL=1\n");
        exit(1);
      #endif
    }
  }
  print_result(info, br);

  if (opts.csv) print_csv_row(info, ct, br);
}
