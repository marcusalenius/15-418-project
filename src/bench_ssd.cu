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
#include "bench_common.h"

#ifdef USE_NCCL
#include <thread>
#include <pthread.h>
#include <atomic>
#include <nccl.h>
#include <nvtx3/nvToolsExt.h>
#include "forward_tp.h"
#endif


#ifdef USE_NCCL

// ---------------------------------------------------------------------------
// Component timing helpers
// ---------------------------------------------------------------------------

// Time one draft forward pass at M tokens on a single GPU.
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

// Time one target verification forward pass (TP-sharded).
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
static BenchResult bench_ssd_e2e_tp(
  const ModelConfig& target_cfg,
  const ModelConfig& draft_cfg,
  int T,
  int K,
  int F,
  int N,
  float alpha,
  float phit,
  int warmup_iters,
  int bench_iters
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
      // match bench_sd_e2e_tp).
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

      // NVTX range so `nsys --capture-range=nvtx -p bench@*` captures only
      // this timed section (skipping allocation, warmup, etc.). Rank 0 alone
      // pushes the range; nsys records all GPUs' activity during this window.
      if (r == 0) nvtxRangePushA("bench");

      CHECK_CUDA(cudaEventRecord(start, compute_stream));

      for (int iter = 0; iter < bench_iters; iter++) {
        int local_tokens = 0;
        if (!is_draft) pthread_barrier_wait(&target_sync);
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

      if (r == 0) nvtxRangePop();

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

  BenchResult br;
  br.e2e_ms     = results[0] / bench_iters;
  br.avg_tokens = static_cast<float>(total_tokens) / bench_iters;
  br.avg_rounds = static_cast<float>(total_rounds) / bench_iters;
  br.miss_rate  = (total_rounds > 0)
    ? static_cast<float>(total_misses) / total_rounds
    : 0.0f;
  br.throughput_tok_s = br.e2e_ms > 0
    ? br.avg_tokens * 1000.0f / br.e2e_ms
    : 0.0f;
  return br;
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
  int bench_iters,
  const BenchOpts& opts
) {
  #ifndef USE_NCCL
    (void)target_cfg; (void)draft_cfg; (void)tp; (void)K; (void)F;
    (void)N; (void)alpha; (void)phit; (void)warmup_iters; (void)bench_iters;
    (void)opts;
    fprintf(stderr, "Error: SSD mode requires building with USE_NCCL=1\n");
    exit(1);
  #else

  bool target_fits = model_fits(target_cfg, tp);
  bool draft_fits  = model_fits(draft_cfg, 1);

  BenchInfo info;
  info.mode = "ssd";
  info.target = &target_cfg;
  info.draft  = &draft_cfg;
  info.tp = tp;
  info.K = K;
  info.F = F;
  info.N = N;
  info.alpha = alpha;
  info.phit  = phit;
  info.warmup = warmup_iters;
  info.iters  = bench_iters;
  info.target_fits = target_fits;
  info.draft_fits  = draft_fits;

  print_bench_header(info);

  if (!target_fits || !draft_fits) {
    fprintf(stderr,
      "Error: SSD end-to-end requires both target (TP=%d) and draft to fit in GPU memory.\n",
      tp);
    exit(1);
  }

  // Component timing
  ComponentTimes ct;
  if (!opts.skip_component) {
    int F_eff = F > 0 ? F : 1;
    ct.draft_step_ms = bench_draft_pass_single(draft_cfg, 1, warmup_iters, bench_iters);
    float draft_fanout_ms = (F_eff > 1)
      ? bench_draft_pass_single(draft_cfg, F_eff, warmup_iters, bench_iters)
      : ct.draft_step_ms;
    ct.draft_total_ms   = ct.draft_step_ms * K;
    ct.prespec_ms       = draft_fanout_ms * K;
    ct.target_verify_ms = bench_target_verify_tp(target_cfg, tp, K, warmup_iters, bench_iters);
  }
  print_components(info, ct);

  // End-to-end
  BenchResult br;
  if (!opts.skip_e2e) {
    br = bench_ssd_e2e_tp(target_cfg, draft_cfg, tp, K, F, N, alpha, phit,
                          warmup_iters, bench_iters);
  }
  print_result(info, br);

  if (opts.csv) print_csv_row(info, ct, br);

#endif // USE_NCCL
}
