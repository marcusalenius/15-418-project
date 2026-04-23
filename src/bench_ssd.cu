// bench_ssd.cu
// SSD benchmark implementations.

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
#include <pthread.h>
#include <atomic>
#include <nccl.h>
#include "forward_tp.h"
#endif


#ifdef USE_NCCL

// ---------------------------------------------------------------------------
// Analytical helpers
// ---------------------------------------------------------------------------
static double ssd_expected_tokens(int K, double alpha) {
  if (alpha < 1e-9) return 1.0;
  if (std::abs(alpha - 1.0) < 1e-9) return K + 1.0;
  return (1.0 - std::pow(alpha, K + 1)) / (1.0 - alpha);
}


// ---------------------------------------------------------------------------
// Component timing helpers
// ---------------------------------------------------------------------------

// Time one draft forward pass with the given M on a single GPU
static float bench_draft_pass_single(
  const ModelConfig& draft_cfg,
  int M,
  int warmup_iters,
  int bench_iters
) {
  cudaSetDevice(0);
  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));
  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  int L = draft_cfg.n_layers;
  std::vector<LayerWeights> weights(L);
  for (int l = 0; l < L; l++)
    alloc_layer_weights(weights[l], draft_cfg);
  ActivationBuffers buf = alloc_activations(draft_cfg, M);

  for (int i = 0; i < warmup_iters; i++)
    forward_model(handle, draft_cfg, weights.data(), M,
                  buf.x, buf.scratch1, buf.scratch2);
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < bench_iters; i++)
    forward_model(handle, draft_cfg, weights.data(), M,
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
}

// Time one target verification forward pass (TP-sharded)
static float bench_target_verify_tp(
  const ModelConfig& target_cfg,
  int T,
  int K,
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

      int L = target_cfg.n_layers;
      std::vector<TPLayerWeights> weights(L);
      for (int l = 0; l < L; l++)
        alloc_tp_layer_weights(weights[l], target_cfg, T);
      ActivationBuffers buf = alloc_activations_tp(target_cfg, K, T);

      for (int i = 0; i < warmup_iters; i++)
        forward_model_tp(ctx.handle, ctx.comm, ctx.stream,
                         target_cfg, weights.data(), T, K,
                         buf.x, buf.scratch1, buf.scratch2);
      CHECK_CUDA(cudaStreamSynchronize(ctx.stream));

      cudaEvent_t start, stop;
      CHECK_CUDA(cudaEventCreate(&start));
      CHECK_CUDA(cudaEventCreate(&stop));

      CHECK_CUDA(cudaEventRecord(start, ctx.stream));
      for (int i = 0; i < bench_iters; i++)
        forward_model_tp(ctx.handle, ctx.comm, ctx.stream,
                         target_cfg, weights.data(), T, K,
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

  for (auto& t : threads) t.join();
  return results[0];
}


// ---------------------------------------------------------------------------
// End-to-end SSD benchmark
//
// Launches T+1 threads:
//   - ranks [0..T-1] are the tensor-parallel target
//   - rank  T        is the dedicated draft GPU
//
// NCCL topology: two separate communicators.
//   tp_comm:  size T, ranks [0..T-1]         (used by forward_model_tp)
//   dt_comm:  size 2, [target rank 0, draft] (used for outcome/spec send-recv)
//
// Draft uses two CUDA streams (compute + comm) so that pre-speculation runs
// concurrently with the outcome recv.
// ---------------------------------------------------------------------------
static float bench_ssd_e2e_tp(
  const ModelConfig& target_cfg,
  const ModelConfig& draft_cfg,
  int T,
  int K,
  int F,
  int N,
  float alpha,
  float phit,
  int warmup_iters,
  int bench_iters,
  int* out_total_tokens,
  int* out_total_rounds,
  int* out_total_misses
) {
  int total_ranks = T + 1;

  ncclUniqueId tp_id, dt_id;
  CHECK_NCCL(ncclGetUniqueId(&tp_id));
  CHECK_NCCL(ncclGetUniqueId(&dt_id));

  std::vector<float> results(total_ranks);
  std::vector<std::thread> threads;

  pthread_barrier_t target_sync;
  pthread_barrier_init(&target_sync, nullptr, T);
  std::atomic<bool> round_done{false};

  // Only rank 0 updates these
  int total_tokens = 0;
  int total_rounds = 0;
  // Only draft updates this
  int total_misses = 0;

  int F_eff = F > 0 ? F : 1;

  for (int r = 0; r < total_ranks; r++) {
    threads.emplace_back([&, r]() {
      bool is_draft = (r == T);
      CHECK_CUDA(cudaSetDevice(r));

      // -----------------------------------------------------------------
      // Per-rank handles / streams
      // -----------------------------------------------------------------
      cublasHandle_t handle;
      CHECK_CUBLAS(cublasCreate(&handle));
      CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

      cudaStream_t compute_stream;
      CHECK_CUDA(cudaStreamCreate(&compute_stream));
      CHECK_CUBLAS(cublasSetStream(handle, compute_stream));

      // Only draft needs an independent comm_stream (so pre-spec on
      // compute_stream can overlap with recv on comm_stream). Target rank 0
      // does its dt_comm ops on compute_stream (strictly sequential anyway:
      // recv spec -> verify -> send outcome).
      cudaStream_t comm_stream = nullptr;
      if (is_draft) CHECK_CUDA(cudaStreamCreate(&comm_stream));

      // -----------------------------------------------------------------
      // NCCL comms
      // -----------------------------------------------------------------
      ncclComm_t tp_comm = nullptr;
      ncclComm_t dt_comm = nullptr;

      if (!is_draft) {
        CHECK_NCCL(ncclCommInitRank(&tp_comm, T, tp_id, r));
      }
      if (is_draft) {
        CHECK_NCCL(ncclCommInitRank(&dt_comm, 2, dt_id, 1));
      } else if (r == 0) {
        CHECK_NCCL(ncclCommInitRank(&dt_comm, 2, dt_id, 0));
      }

      // -----------------------------------------------------------------
      // Weights & activations
      // -----------------------------------------------------------------
      std::vector<TPLayerWeights> target_w;
      ActivationBuffers target_buf = {nullptr, nullptr, nullptr};
      std::vector<LayerWeights> draft_w;
      ActivationBuffers draft_buf_F = {nullptr, nullptr, nullptr};
      ActivationBuffers draft_buf_1 = {nullptr, nullptr, nullptr};

      if (is_draft) {
        int dL = draft_cfg.n_layers;
        draft_w.resize(dL);
        for (int l = 0; l < dL; l++)
          alloc_layer_weights(draft_w[l], draft_cfg);
        draft_buf_F = alloc_activations(draft_cfg, F_eff);
        draft_buf_1 = alloc_activations(draft_cfg, 1);
      } else {
        int tL = target_cfg.n_layers;
        target_w.resize(tL);
        for (int l = 0; l < tL; l++)
          alloc_tp_layer_weights(target_w[l], target_cfg, T);
        target_buf = alloc_activations_tp(target_cfg, K, T);
      }

      // NCCL buffers
      // outcome: 3 ints  = [acc_len, bonus_id, done_flag]
      // spec:    K halfs (dummy)
      int*  outcome_buf = nullptr;
      half* spec_buf    = nullptr;
      if (is_draft || r == 0) {
        CHECK_CUDA(cudaMalloc(&outcome_buf, 3 * sizeof(int)));
        CHECK_CUDA(cudaMemset(outcome_buf, 0, 3 * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&spec_buf, K * sizeof(half)));
        CHECK_CUDA(cudaMemset(spec_buf, 0x3C, K * sizeof(half)));
      }

      // -----------------------------------------------------------------
      // RNG
      // Target rank 0 samples Bernoulli(alpha) for acceptance (seed 42 to 
      // match bench_sd_e2e_tp)
      // Draft samples Bernoulli(phit) for cache hit.
      // -----------------------------------------------------------------
      std::mt19937 rng(is_draft ? 1337 : 42);
      std::bernoulli_distribution accept_dist(alpha);
      std::bernoulli_distribution hit_dist(phit);

      // -----------------------------------------------------------------
      // Warmup
      // -----------------------------------------------------------------
      for (int i = 0; i < warmup_iters; i++) {
        if (is_draft) {
          for (int k = 0; k < K; k++)
            forward_model(handle, draft_cfg, draft_w.data(), 1,
                          draft_buf_1.x, draft_buf_1.scratch1, draft_buf_1.scratch2);
          for (int k = 0; k < K; k++)
            forward_model(handle, draft_cfg, draft_w.data(), F_eff,
                          draft_buf_F.x, draft_buf_F.scratch1, draft_buf_F.scratch2);
          CHECK_CUDA(cudaStreamSynchronize(compute_stream));
        } else {
          forward_model_tp(handle, tp_comm, compute_stream,
                           target_cfg, target_w.data(), T, K,
                           target_buf.x, target_buf.scratch1, target_buf.scratch2);
          CHECK_CUDA(cudaStreamSynchronize(compute_stream));
        }
      }

      // -----------------------------------------------------------------
      // Timed region
      // -----------------------------------------------------------------
      cudaEvent_t start, stop;
      CHECK_CUDA(cudaEventCreate(&start));
      CHECK_CUDA(cudaEventCreate(&stop));

      int host_outcome[3] = {0, 0, 0};
      int local_misses = 0;

      CHECK_CUDA(cudaEventRecord(start, compute_stream));

      for (int iter = 0; iter < bench_iters; iter++) {
        int local_tokens = 0;
        if (r == 0) round_done.store(false);
        if (!is_draft) pthread_barrier_wait(&target_sync);

        if (is_draft) {
          // -----------------------------------------------------------
          // Just-in-time speculation: no pre-spec available for round 0
          // -----------------------------------------------------------
          for (int k = 0; k < K; k++)
            forward_model(handle, draft_cfg, draft_w.data(), 1,
                          draft_buf_1.x, draft_buf_1.scratch1, draft_buf_1.scratch2);
          CHECK_CUDA(cudaStreamSynchronize(compute_stream));

          CHECK_NCCL(ncclSend(spec_buf, K, ncclFloat16, 0, dt_comm, comm_stream));
          CHECK_CUDA(cudaStreamSynchronize(comm_stream));

          // -----------------------------------------------------------
          // Steady-state
          // -----------------------------------------------------------
          while (true) {
            // Pre-speculation for the next round's spec, launched async
            // on compute_stream. This overlaps with target's verification
            // of the spec we just sent. Batched M=F drafts F candidate
            // branches in parallel.
            for (int k = 0; k < K; k++)
              forward_model(handle, draft_cfg, draft_w.data(), F_eff,
                            draft_buf_F.x, draft_buf_F.scratch1, draft_buf_F.scratch2);

            // Recv outcome from target on comm_stream (independent of
            // compute_stream so it does not serialize with pre-spec)
            CHECK_NCCL(ncclRecv(outcome_buf, 3, ncclInt32, 0, dt_comm, comm_stream));
            CHECK_CUDA(cudaStreamSynchronize(comm_stream));
            CHECK_CUDA(cudaMemcpy(host_outcome, outcome_buf,
                                  3 * sizeof(int), cudaMemcpyDeviceToHost));

            bool is_done = host_outcome[2] != 0;
            if (is_done) {
              // Drain pending pre-spec before exiting so the next bench
              // iter starts from a clean stream
              CHECK_CUDA(cudaStreamSynchronize(compute_stream));
              break;
            }

            // Wait for pre-speculation compute to finish before deciding
            // hit/miss
            CHECK_CUDA(cudaStreamSynchronize(compute_stream));

            bool cache_hit = hit_dist(rng);
            if (!cache_hit) {
              local_misses++;
              // Fallback: the pre-speculation did not match the
              // target's outcome, so run the just-in-time draft
              for (int k = 0; k < K; k++)
                forward_model(handle, draft_cfg, draft_w.data(), 1,
                              draft_buf_1.x, draft_buf_1.scratch1, draft_buf_1.scratch2);
              CHECK_CUDA(cudaStreamSynchronize(compute_stream));
            }

            CHECK_NCCL(ncclSend(spec_buf, K, ncclFloat16, 0, dt_comm, comm_stream));
            CHECK_CUDA(cudaStreamSynchronize(comm_stream));
          }
        } else {
          // -----------------------------------------------------------
          // Target rank (any of 0..T-1)
          // -----------------------------------------------------------
          while (true) {
            // Rank 0: receive the speculation from draft (the subsequent 
            // forward_model_tp all-reduce will wait for it to complete via 
            // stream ordering)
            if (r == 0) {
              CHECK_NCCL(ncclRecv(spec_buf, K, ncclFloat16, 1, dt_comm, compute_stream));
            }
            pthread_barrier_wait(&target_sync);

            // All target ranks verify
            forward_model_tp(handle, tp_comm, compute_stream,
                             target_cfg, target_w.data(), T, K,
                             target_buf.x, target_buf.scratch1, target_buf.scratch2);
            CHECK_CUDA(cudaStreamSynchronize(compute_stream));

            if (r == 0) {
              // Synthetic Bernoulli(alpha) acceptance
              int accepted = 0;
              for (int k = 0; k < K; k++) {
                if (accept_dist(rng))
                  accepted++;
                else
                  break;
              }
              local_tokens += accepted + 1;
              total_rounds++;

              int is_done = (local_tokens >= N) ? 1 : 0;
              if (is_done) {
                total_tokens += local_tokens;
                round_done.store(true);
              }

              int outcome_host[3] = {accepted, 0, is_done};
              CHECK_CUDA(cudaMemcpyAsync(outcome_buf, outcome_host, 3 * sizeof(int),
                                         cudaMemcpyHostToDevice, compute_stream));
              CHECK_NCCL(ncclSend(outcome_buf, 3, ncclInt32, 1, dt_comm, compute_stream));
              CHECK_CUDA(cudaStreamSynchronize(compute_stream));
            }
            pthread_barrier_wait(&target_sync);

            if (round_done.load()) break;
          }
        }
      }

      CHECK_CUDA(cudaEventRecord(stop, compute_stream));
      CHECK_CUDA(cudaEventSynchronize(stop));

      float total_ms = 0;
      CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
      results[r] = total_ms;

      if (is_draft) total_misses += local_misses;

      // -----------------------------------------------------------------
      // Cleanup
      // -----------------------------------------------------------------
      CHECK_CUDA(cudaEventDestroy(start));
      CHECK_CUDA(cudaEventDestroy(stop));

      if (is_draft) {
        free_activations(draft_buf_F);
        free_activations(draft_buf_1);
        for (int l = 0; l < (int)draft_w.size(); l++)
          free_layer_weights(draft_w[l]);
      } else {
        free_activations(target_buf);
        for (int l = 0; l < (int)target_w.size(); l++)
          free_tp_layer_weights(target_w[l]);
      }
      if (outcome_buf) cudaFree(outcome_buf);
      if (spec_buf)    cudaFree(spec_buf);
      if (comm_stream) CHECK_CUDA(cudaStreamDestroy(comm_stream));
      CHECK_CUDA(cudaStreamDestroy(compute_stream));
      CHECK_CUBLAS(cublasDestroy(handle));
      if (tp_comm) ncclCommDestroy(tp_comm);
      if (dt_comm) ncclCommDestroy(dt_comm);
    });
  }

  for (auto& t : threads) t.join();
  pthread_barrier_destroy(&target_sync);

  if (out_total_tokens) *out_total_tokens = total_tokens;
  if (out_total_rounds) *out_total_rounds = total_rounds;
  if (out_total_misses) *out_total_misses = total_misses;

  // Return the target-rank-0 wall-clock
  return results[0];
}

#endif // USE_NCCL


// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void run_ssd_benchmark(
  const ModelConfig& target_cfg,
  const ModelConfig& draft_cfg,
  int tp,
  int K,
  int F,
  int N,
  float alpha,
  float phit,
  int warmup_iters,
  int bench_iters
) {
#ifndef USE_NCCL
  (void)target_cfg; (void)draft_cfg; (void)tp; (void)K; (void)F;
  (void)N; (void)alpha; (void)phit; (void)warmup_iters; (void)bench_iters;
  fprintf(stderr, "Error: SSD mode requires building with USE_NCCL=1\n");
  exit(1);
#else

  bool target_fits = model_fits(target_cfg, tp);
  bool draft_fits  = model_fits(draft_cfg, 1);

  printf("--- SSD Benchmark ---\n");
  printf("  Target: %s (d=%d, L=%d)\n",
         target_cfg.name.c_str(), target_cfg.d_model, target_cfg.n_layers);
  printf("  Draft:  %s (d=%d, L=%d)\n",
         draft_cfg.name.c_str(), draft_cfg.d_model, draft_cfg.n_layers);
  printf("  TP=%d (+1 draft GPU), K=%d, F=%d, alpha=%.2f, phit=%.2f, N=%d\n\n",
         tp, K, F, alpha, phit, N);

  size_t target_bytes = (tp == 1)
    ? target_cfg.total_weight_bytes()
    : target_cfg.total_tp_weight_bytes(tp);
  size_t draft_bytes = draft_cfg.total_weight_bytes();
  printf("  Target weight footprint%s: %.2f GB | Fits: %s\n",
         tp > 1 ? " (per GPU)" : "",
         target_bytes / (1024.0 * 1024 * 1024),
         target_fits ? "YES" : "NO");
  printf("  Draft weight footprint:       %.2f GB | Fits: %s\n\n",
         draft_bytes / (1024.0 * 1024 * 1024),
         draft_fits ? "YES" : "NO");

  if (!target_fits || !draft_fits) {
    fprintf(stderr,
      "Error: SSD end-to-end requires both target (TP=%d) and draft to fit in GPU memory.\n",
      tp);
    exit(1);
  }

  // -----------------------------------------------------------------------
  // Component timing
  // -----------------------------------------------------------------------
  printf("  [Component Timing]\n");

  float draft_step_ms   = bench_draft_pass_single(draft_cfg, 1, warmup_iters, bench_iters);
  float draft_fanout_ms = (F > 1)
    ? bench_draft_pass_single(draft_cfg, F, warmup_iters, bench_iters)
    : draft_step_ms;
  float draft_total_ms  = draft_step_ms * K;
  float prespec_ms      = draft_fanout_ms * K;

  float target_verify_ms =
    bench_target_verify_tp(target_cfg, tp, K, warmup_iters, bench_iters);

  float t_sd_ratio = draft_total_ms / target_verify_ms;

  printf("    Draft step (M=1):            %.3f ms\n", draft_step_ms);
  printf("    Draft step (M=%d):           %.3f ms\n", F, draft_fanout_ms);
  printf("    Draft total (K=%d, M=1):     %.3f ms  (SD draft / fallback)\n",
         K, draft_total_ms);
  printf("    Draft pre-spec (K=%d, M=%d): %.3f ms  (SSD pipeline work)\n",
         K, F, prespec_ms);
  printf("    Target verify (M=%d):        %.3f ms\n", K, target_verify_ms);
  printf("    T_SD ratio (T_draft/T_tgt):  %.4f\n", t_sd_ratio);
  printf("    Pre-spec fits in verify?     %s (%.3f ms budget, %.3f ms used)\n\n",
         (prespec_ms <= target_verify_ms) ? "YES" : "NO",
         target_verify_ms, prespec_ms);

  // -----------------------------------------------------------------------
  // Analytical prediction
  //
  // Per-round time normalized to T_target:
  //   SD : 1 + T_SD
  //   SSD: phit * max(1, T_p) + (1-phit) * (1 + T_b)
  //   where T_p = pre-spec / T_target,  T_b = draft / T_target
  // -----------------------------------------------------------------------
  double t_p = prespec_ms / target_verify_ms;
  double t_b = draft_total_ms / target_verify_ms;
  double sd_round_norm  = 1.0 + t_sd_ratio;
  double ssd_round_norm = phit * std::max(1.0, t_p)
                        + (1.0 - phit) * (1.0 + t_b);
  double ssd_over_sd    = sd_round_norm / ssd_round_norm;

  printf("  [Analytical Prediction]\n");
  printf("    SD  per-round (norm):  %.3f\n", sd_round_norm);
  printf("    SSD per-round (norm):  %.3f\n", ssd_round_norm);
  printf("    SSD speedup over SD:   %.2fx (upper bound 1 + T_SD = %.2fx at phit=1)\n\n",
         ssd_over_sd, 1.0 + t_sd_ratio);

  // -----------------------------------------------------------------------
  // End-to-end run
  // -----------------------------------------------------------------------
  printf("  [End-to-End Simulation] (alpha=%.2f, phit=%.2f, N=%d)\n",
         alpha, phit, N);

  int total_tokens = 0, total_rounds = 0, total_misses = 0;
  float e2e_ms = bench_ssd_e2e_tp(
    target_cfg, draft_cfg, tp, K, F, N, alpha, phit,
    warmup_iters, bench_iters,
    &total_tokens, &total_rounds, &total_misses
  );

  float avg_ms     = e2e_ms / bench_iters;
  float avg_tokens = static_cast<float>(total_tokens) / bench_iters;
  float avg_rounds = static_cast<float>(total_rounds) / bench_iters;
  float miss_rate  = (total_rounds > 0)
    ? static_cast<float>(total_misses) / total_rounds
    : 0.0f;

  printf("    Avg tokens generated: %.1f (target %d)\n", avg_tokens, N);
  printf("    Avg rounds:           %.1f\n", avg_rounds);
  printf("    Avg tokens/round:     %.2f\n", avg_tokens / avg_rounds);
  printf("    Cache miss rate:      %.2f  (expected 1 - phit = %.2f)\n",
         miss_rate, 1.0f - phit);
  printf("    Total time:           %.2f ms\n", avg_ms);
  printf("    Throughput:           %.1f tok/s\n",
         avg_tokens * 1000.0f / avg_ms);

  // -----------------------------------------------------------------------
  // Correctness check: under phit=0, SSD degenerates to a pipeline where
  // every round falls back. The token-generation process is identical to
  // SD (same acceptance distribution), so avg tokens/round should equal
  // E_SD = (1 - alpha^(K+1)) / (1 - alpha).
  // -----------------------------------------------------------------------
  double expected = ssd_expected_tokens(K, alpha);
  float  measured = avg_tokens / avg_rounds;
  float  rel_err  = std::abs(measured - expected) / expected;

  if (phit < 1e-6) {
    printf("\n  [Correctness @ phit=0]\n");
    printf("    E[tokens/round] (SD): %.3f\n", expected);
    printf("    Measured:             %.3f\n", measured);
    printf("    Relative error:       %.2f%%  %s\n",
           rel_err * 100,
           rel_err < 0.05f ? "(OK, within 5%)" : "(WARN: >5% drift)");
  }

  // -----------------------------------------------------------------------
  // Summary
  // -----------------------------------------------------------------------
  float sd_round_ms      = draft_total_ms + target_verify_ms;
  float sd_throughput    = static_cast<float>(expected) * 1000.0f / sd_round_ms;
  float ar_throughput    = 0.0f;  // Not measured here; see bench_ar.
  float ssd_throughput   = avg_tokens * 1000.0f / avg_ms;
  float ssd_vs_sd        = ssd_throughput / sd_throughput;

  printf("\n  [Summary] (alpha=%.2f, K=%d, F=%d, phit=%.2f)\n",
         alpha, K, F, phit);
  printf("    E[tokens/round]:  %.2f\n", expected);
  printf("    SD  throughput:   %.1f tok/s (%.3f ms/round)\n",
         sd_throughput, sd_round_ms);
  printf("    SSD throughput:   %.1f tok/s (%.3f ms/round)\n",
         ssd_throughput, avg_ms / avg_rounds);
  printf("    SSD speedup:      %.2fx over SD\n\n", ssd_vs_sd);

  (void)ar_throughput;
#endif // USE_NCCL
}
