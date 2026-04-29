// bench_common.cu
// Implementations of the uniform print and CSV helpers declared in
// bench_common.h.

#include <cstdio>
#include "bench_common.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static const char* fits_str(bool fits) {
  return fits ? "YES" : "NO (layer estimate)";
}

static double gb(size_t bytes) {
  return bytes / (1024.0 * 1024.0 * 1024.0);
}

// ---------------------------------------------------------------------------
// Header
// ---------------------------------------------------------------------------

void print_bench_header(const BenchInfo& info) {
  const ModelConfig& t = *info.target;
  printf("--- %s Benchmark ---\n",
         info.mode == "ar"  ? "AR" :
         info.mode == "sd"  ? "SD" :
         info.mode == "ssd" ? "SSD" : info.mode.c_str());
  printf("  Target: %s (d=%d, L=%d)\n", t.name.c_str(), t.d_model, t.n_layers);
  if (info.draft) {
    const ModelConfig& d = *info.draft;
    printf("  Draft:  %s (d=%d, L=%d)\n",
           d.name.c_str(), d.d_model, d.n_layers);
  }

  // Config echo
  if (info.mode == "ar") {
    printf("  TP=%d, N=%d\n", info.tp, info.N);
  } else if (info.mode == "sd") {
    printf("  TP=%d, K=%d, alpha=%.2f, N=%d\n",
           info.tp, info.K, info.alpha, info.N);
  } else if (info.mode == "ssd") {
    printf("  TP=%d (+1 draft GPU), K=%d, F=%d, alpha=%.2f, phit=%.2f, N=%d\n",
           info.tp, info.K, info.F, info.alpha, info.phit, info.N);
  }

  // Fit info
  size_t target_bytes = (info.tp == 1)
    ? t.total_weight_bytes()
    : t.total_tp_weight_bytes(info.tp);
  printf("  Target weight footprint%s: %.2f GB | Fits: %s\n",
         info.tp > 1 ? " (per GPU)" : "",
         gb(target_bytes), fits_str(info.target_fits));
  if (info.draft) {
    printf("  Draft weight footprint:       %.2f GB | Fits: %s\n",
           gb(info.draft->total_weight_bytes()),
           fits_str(info.draft_fits));
  }
  printf("\n");
}

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

void print_components(const BenchInfo& info, const ComponentTimes& ct) {
  // If everything is zero the caller skipped this block.
  if (ct.target_step_ms == 0.0f && ct.target_verify_ms == 0.0f &&
      ct.draft_step_ms  == 0.0f && ct.draft_total_ms   == 0.0f &&
      ct.prespec_ms     == 0.0f) {
    return;
  }

  printf("  [Component Timing]\n");
  if (ct.target_step_ms > 0.0f)
    printf("    Target step  (M=1):       %.3f ms\n", ct.target_step_ms);
  if (ct.target_verify_ms > 0.0f)
    printf("    Target verify(M=%d):       %.3f ms\n",
           info.K, ct.target_verify_ms);
  if (ct.draft_step_ms > 0.0f)
    printf("    Draft step   (M=1):       %.3f ms\n", ct.draft_step_ms);
  if (ct.draft_total_ms > 0.0f)
    printf("    Draft total  (K=%d, M=1):  %.3f ms\n",
           info.K, ct.draft_total_ms);
  if (ct.prespec_ms > 0.0f)
    printf("    Pre-spec     (K=%d, M=%d): %.3f ms\n",
           info.K, info.F, ct.prespec_ms);
  printf("\n");
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

void print_result(const BenchInfo& info, const BenchResult& br) {
  if (br.e2e_ms == 0.0f) return;

  printf("  [End-to-End]\n");
  printf("    Avg tokens:   %.1f\n", br.avg_tokens);
  printf("    Avg rounds:   %.1f\n", br.avg_rounds);
  if (br.avg_rounds > 0.0f)
    printf("    Tokens/round: %.2f\n", br.avg_tokens / br.avg_rounds);
  if (info.mode == "ssd")
    printf("    Miss rate:    %.2f  (expected 1 - phit = %.2f)\n",
           br.miss_rate, 1.0f - info.phit);
  printf("    E2E time:     %.2f ms\n", br.e2e_ms);
  printf("    Throughput:   %.1f tok/s\n\n", br.throughput_tok_s);
}

// ---------------------------------------------------------------------------
// CSV
// ---------------------------------------------------------------------------

// Schema:
//   tag, mode, target, draft, tp, K, F, N, alpha, phit,
//   target_fits, draft_fits,
//   target_step_ms, target_verify_ms, draft_step_ms, draft_total_ms, prespec_ms,
//   e2e_ms, avg_tokens, avg_rounds, miss_rate, throughput_tok_s
//
// target_fits / draft_fits are 0/1 flags. When 0, the timing columns for
// that model are layer-extrapolated (one layer's forward pass scaled to
// n_layers) rather than a true full-model measurement.
void print_csv_header() {
  printf("tag,mode,target,draft,tp,K,F,N,alpha,phit,"
         "target_fits,draft_fits,"
         "target_step_ms,target_verify_ms,draft_step_ms,draft_total_ms,prespec_ms,"
         "e2e_ms,avg_tokens,avg_rounds,miss_rate,throughput_tok_s\n");
}

void print_csv_row(const BenchInfo& info,
                   const ComponentTimes& ct,
                   const BenchResult& br) {
  const char* target_name = info.target ? info.target->name.c_str() : "";
  const char* draft_name  = info.draft  ? info.draft->name.c_str()  : "";
  printf(
    "RESULT,%s,%s,%s,%d,%d,%d,%d,%.4f,%.4f,"
    "%d,%d,"
    "%.4f,%.4f,%.4f,%.4f,%.4f,"
    "%.4f,%.4f,%.4f,%.4f,%.4f\n",
    info.mode.c_str(), target_name, draft_name,
    info.tp, info.K, info.F, info.N, info.alpha, info.phit,
    info.target_fits ? 1 : 0, info.draft_fits ? 1 : 0,
    ct.target_step_ms, ct.target_verify_ms,
    ct.draft_step_ms, ct.draft_total_ms, ct.prespec_ms,
    br.e2e_ms, br.avg_tokens, br.avg_rounds,
    br.miss_rate, br.throughput_tok_s
  );
}
