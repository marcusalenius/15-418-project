// main.cu

#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "config.h"
#include "bench_common.h"
#include "bench_ar.h"
#include "bench_sd.h"
#include "bench_ssd.h"

// ---------------------------------------------------------------------------
// CLI helpers
// ---------------------------------------------------------------------------
static void print_usage(const std::string& prog) {
  printf(
    "Usage: %s [options]\n"
    "\n"
    "Options:\n"
    "  --target-model <name>   Target model (llama-1b, llama-8b, llama-70b)  [llama-8b]\n"
    "  --draft-model  <name>   Draft model  (llama-1b, llama-8b, llama-70b)  [llama-1b]\n"
    "  --tp           <int>    Tensor parallelism degree                     [1]\n"
    "  --mode         <str>    Inference mode (ar, sd, ssd)                  [ar]\n"
    "  --K            <int>    Speculation length (sd/ssd only)              [4]\n"
    "  --F            <int>    Fan-out for pre-speculation (ssd only)        [4]\n"
    "  --alpha        <float>  Acceptance rate (sd/ssd only)                 [0.9]\n"
    "  --phit         <float>  Cache hit probability (ssd only)              [0.85]\n"
    "  --N            <int>    Tokens to generate                            [128]\n"
    "  --warmup       <int>    Warmup iterations                             [10]\n"
    "  --iters        <int>    Benchmark iterations                          [20]\n"
    "  --skip-component        Skip per-stage forward-pass timing\n"
    "  --skip-e2e              Skip the end-to-end simulation\n"
    "  --csv                   Emit one CSV row at the end (prefixed RESULT,)\n"
    "  --csv-header            Print CSV header line and exit\n"
    "  --help                  Show this message\n",
    prog.c_str()
  );
}

static std::string require_arg(
  int argc,
  char** argv,
  int i,
  const std::string& flag
) {
  if (i + 1 >= argc) {
    fprintf(stderr, "Error: %s requires an argument\n", flag.c_str());
    exit(1);
  }
  return argv[i + 1];
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

static void validate_tp(
  const ModelConfig& model,
  int tp,
  const std::string& role
) {
  if (!model.divisible_by(tp)) {
    fprintf(stderr,
      "Error: %s model '%s' dimensions not evenly divisible by TP=%d\n"
      "  q_dim=%d, kv_dim=%d, d_ff=%d\n",
      role.c_str(), model.name.c_str(), tp,
      model.q_dim(), model.kv_dim(), model.d_ff);
    exit(1);
  }
}

static void validate_gpu_count(int tp, const std::string& mode) {
  int num_gpus;
  cudaGetDeviceCount(&num_gpus);

  int required = tp;
  if (mode == "ssd") required = tp + 1;

  if (required > num_gpus) {
    fprintf(stderr,
      "Error: mode '%s' with TP=%d requires %d GPU(s), but only %d available\n",
      mode.c_str(), tp, required, num_gpus);
    exit(1);
  }
}

// ---------------------------------------------------------------------------
// GPU info
// ---------------------------------------------------------------------------

static void print_gpu_info(int num_gpus) {
  printf("GPUs:\n");
  for (int i = 0; i < num_gpus; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    cudaSetDevice(i);
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf(
      "  [%d] %s | %.2f GB free / %.2f GB total | %d SMs\n",
      i, prop.name,
      free_mem / (1024.0 * 1024 * 1024),
      total_mem / (1024.0 * 1024 * 1024),
      prop.multiProcessorCount
    );
  }
  printf("\n");
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  // Defaults
  std::string target_model_name = "llama-8b";
  std::string draft_model_name = "llama-1b";
  std::string mode = "ar";
  int tp = 1;
  int K = 4;
  int F = 4;
  float alpha = 0.9f;
  float phit = 0.85f;
  int N = 128;
  int warmup = 10;
  int iters = 20;
  BenchOpts opts;
  bool csv_header_only = false;

  // Parse arguments
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    } else if (arg == "--target-model") {
      target_model_name = require_arg(argc, argv, i, arg); i++;
    } else if (arg == "--draft-model") {
      draft_model_name = require_arg(argc, argv, i, arg); i++;
    } else if (arg == "--tp") {
      tp = std::stoi(require_arg(argc, argv, i, arg)); i++;
    } else if (arg == "--mode") {
      mode = require_arg(argc, argv, i, arg); i++;
    } else if (arg == "--K") {
      K = std::stoi(require_arg(argc, argv, i, arg)); i++;
    } else if (arg == "--F") {
      F = std::stoi(require_arg(argc, argv, i, arg)); i++;
    } else if (arg == "--alpha") {
      alpha = std::stof(require_arg(argc, argv, i, arg)); i++;
    } else if (arg == "--phit") {
      phit = std::stof(require_arg(argc, argv, i, arg)); i++;
    } else if (arg == "--N") {
      N = std::stoi(require_arg(argc, argv, i, arg)); i++;
    } else if (arg == "--warmup") {
      warmup = std::stoi(require_arg(argc, argv, i, arg)); i++;
    } else if (arg == "--iters") {
      iters = std::stoi(require_arg(argc, argv, i, arg)); i++;
    } else if (arg == "--skip-component") {
      opts.skip_component = true;
    } else if (arg == "--skip-e2e") {
      opts.skip_e2e = true;
    } else if (arg == "--csv") {
      opts.csv = true;
    } else if (arg == "--csv-header") {
      csv_header_only = true;
    } else {
      fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
      print_usage(argv[0]);
      return 1;
    }
  }

  // --csv-header is a utility flag: print and exit, no GPU work.
  if (csv_header_only) {
    print_csv_header();
    return 0;
  }

  // Validate mode
  if (mode != "ar" && mode != "sd" && mode != "ssd") {
    fprintf(stderr, "Error: --mode must be one of: ar, sd, ssd (got '%s')\n", mode.c_str());
    return 1;
  }
  // Validate tp
  if (tp < 1) {
    fprintf(stderr, "Error: --tp must be >= 1\n");
    return 1;
  }

  // Resolve models
  const ModelConfig target_model = lookup_model(target_model_name);
  const ModelConfig draft_model = lookup_model(draft_model_name);

  // Validate hardware
  validate_gpu_count(tp, mode);
  if (tp > 1) validate_tp(target_model, tp, "target");
  if (mode == "sd" || mode == "ssd") {
    if (tp > 1) validate_tp(draft_model, tp, "draft");
  }

  // Print env info
  int num_gpus;
  cudaGetDeviceCount(&num_gpus);
  print_gpu_info(num_gpus);

  // Dispatch
  if (mode == "ar")
    run_ar_benchmark(target_model, tp, N, warmup, iters, opts);
  else if (mode == "sd")
    run_sd_benchmark(target_model, draft_model, tp, K, N, alpha,
                     warmup, iters, opts);
  else if (mode == "ssd")
    run_ssd_benchmark(target_model, draft_model, tp, K, F, N, alpha, phit,
                      warmup, iters, opts);

  return 0;
}
