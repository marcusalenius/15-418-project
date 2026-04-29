// bench_common.h
// Shared types and printing helpers for all benchmark modes.
//
// Every mode (ar/sd/ssd) populates the same two result structs:
//   - ComponentTimes: per-stage forward-pass measurements
//   - BenchResult:    end-to-end measurement
// Fields that don't apply to a given mode are left at 0.
//
// Printing goes through a single set of functions so all modes produce
// identical layouts and the CSV output schema is shared.

#pragma once

#include <string>
#include "config.h"

// ---------------------------------------------------------------------------
// CLI-level options that affect what work the benchmark does and how it
// prints.
// ---------------------------------------------------------------------------
struct BenchOpts {
  bool csv            = false;  // emit one CSV row at the end
  bool skip_component = false;  // skip per-stage forward-pass timings
  bool skip_e2e       = false;  // skip the end-to-end simulation
};

// ---------------------------------------------------------------------------
// Run configuration captured for printing/CSV. Mode-irrelevant fields are
// left at their defaults (e.g. AR has no draft, K, F, alpha, phit).
// ---------------------------------------------------------------------------
struct BenchInfo {
  std::string mode; 
  const ModelConfig* target = nullptr;
  const ModelConfig* draft  = nullptr;
  int   tp     = 1;
  int   K      = 0;
  int   F      = 0;
  int   N      = 0;
  float alpha  = 0.0f;
  float phit   = 0.0f;
  int   warmup = 0;
  int   iters  = 0;
  bool  target_fits = true;
  bool  draft_fits  = true;
};

// ---------------------------------------------------------------------------
// Per-stage forward-pass timings (ms). All fields default to 0 for stages
// that don't apply to the current mode.
// ---------------------------------------------------------------------------
struct ComponentTimes {
  float target_step_ms   = 0.0f;  // one target forward at M=1
  float target_verify_ms = 0.0f;  // one target forward at M=K   (sd/ssd)
  float draft_step_ms    = 0.0f;  // one draft  forward at M=1   (sd/ssd)
  float draft_total_ms   = 0.0f;  // K * draft_step_ms           (sd/ssd)
  float prespec_ms       = 0.0f;  // K * draft forward at M=F    (ssd)
};

// ---------------------------------------------------------------------------
// End-to-end measurement.
//   avg_rounds is 1 for AR (one token per "round"), >=1 for SD/SSD.
//   miss_rate is 0 for AR/SD, in [0,1] for SSD.
// ---------------------------------------------------------------------------
struct BenchResult {
  float e2e_ms           = 0.0f;  // wall-clock per outer iter
  float avg_tokens       = 0.0f;  // tokens generated per iter
  float avg_rounds       = 0.0f;  // rounds per iter
  float throughput_tok_s = 0.0f;
  float miss_rate        = 0.0f;
};

// ---------------------------------------------------------------------------
// Printing
// ---------------------------------------------------------------------------

// Mode header + model info + fit summary.
void print_bench_header(const BenchInfo& info);

// Per-stage forward-pass timings. Skips the block entirely if all fields
// are zero (i.e. component timing was skipped).
void print_components(const BenchInfo& info, const ComponentTimes& ct);

// End-to-end summary. Skips the block if e2e_ms == 0.
void print_result(const BenchInfo& info, const BenchResult& br);

// ---------------------------------------------------------------------------
// CSV output
// ---------------------------------------------------------------------------

// Print just the CSV header line and exit. Useful for driver scripts:
//   ./benchmark --csv-header > results.csv
//   for ... ; do ./benchmark ... --csv | grep '^RESULT,' >> results.csv ; done
void print_csv_header();

// Print one CSV data row prefixed with "RESULT," so it can be grepped out
// of mixed human-readable output.
void print_csv_row(const BenchInfo& info,
                   const ComponentTimes& ct,
                   const BenchResult& br);
