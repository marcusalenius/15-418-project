// config.h

#pragma once

#include <string>
#include <unordered_map>
#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Model config struct
// ---------------------------------------------------------------------------
struct ModelConfig {
  std::string name;
  int d_model;
  int n_heads;
  int n_kv_heads;
  int head_dim;
  int d_ff;
  int n_layers;

  int q_dim()  const { return n_heads * head_dim; }
  int kv_dim() const { return n_kv_heads * head_dim; }

  // Total weight bytes for one layer (FP16)
  size_t layer_weight_bytes() const {
    size_t d   = static_cast<size_t>(d_model);
    size_t dq  = static_cast<size_t>(q_dim());
    size_t dkv = static_cast<size_t>(kv_dim());
    size_t dff = static_cast<size_t>(d_ff);

    return 2 * (       // 2 bytes per FP16 element
      d * dq   +       // W_q
      d * dkv  +       // W_k
      d * dkv  +       // W_v
      dq * d   +       // W_o
      d * dff  +       // W_gate
      d * dff  +       // W_up
      dff * d          // W_down
    );
  }

  // Total weight bytes for one layer sharded across T GPUs
  size_t tp_layer_weight_bytes(int T) const {
    size_t d   = static_cast<size_t>(d_model);
    size_t dq  = static_cast<size_t>(q_dim() / T);
    size_t dkv = static_cast<size_t>(kv_dim() / T);
    size_t dff = static_cast<size_t>(d_ff / T);

    return 2 * (       // 2 bytes per FP16 element
      d * dq   +       // W_q
      d * dkv  +       // W_k
      d * dkv  +       // W_v
      dq * d   +       // W_o
      d * dff  +       // W_gate
      d * dff  +       // W_up
      dff * d          // W_down
    );
  }

  // Total weight bytes for the full model
  size_t total_weight_bytes() const {
    return layer_weight_bytes() * n_layers;
  }

  // Total weight bytes per GPU for TP-sharded full model
  size_t total_tp_weight_bytes(int T) const {
    return tp_layer_weight_bytes(T) * n_layers;
  }

  // Arithmetic intensity of one layer (FLOPs / bytes)
  double arith_intensity(int M) const {
    double d   = static_cast<double>(d_model);
    double dq  = static_cast<double>(q_dim());
    double dkv = static_cast<double>(kv_dim());
    double dff = static_cast<double>(d_ff);
    double m   = static_cast<double>(M);

    double flops = 0;
    double bytes = 0;

    // {M, N, K} for each of the 7 GEMMs
    double gemm_dims[][3] = {
      {m, dq, d}, {m, dkv, d}, {m, dkv, d}, {m, d, dq},
      {m, dff, d}, {m, dff, d}, {m, d, dff}
    };

    for (auto& g : gemm_dims) {
      double gm = g[0], gn = g[1], gk = g[2];
      flops += 2.0 * gm * gn * gk;
      bytes += (gm * gk + gk * gn + gm * gn) * 2.0; // FP16
    }

    return flops / bytes;
  }

  // Check if dimensions are evenly divisible by TP degree
  bool divisible_by(int T) const {
    return q_dim() % T == 0 && kv_dim() % T == 0 && d_ff % T == 0;
  }
};

// ---------------------------------------------------------------------------
// Model registry
// ---------------------------------------------------------------------------
inline const std::unordered_map<std::string, ModelConfig>& model_registry() {
  static const std::unordered_map<std::string, ModelConfig> models = {
    {"llama-1b",  {"Llama-3.2-1B",  2048, 32, 8,  64,  8192, 16}},
    {"llama-8b",  {"Llama-3.1-8B",  4096, 32, 8, 128, 14336, 32}},
    {"llama-70b", {"Llama-3.1-70B", 8192, 64, 8, 128, 28672, 80}},
  };
  return models;
}

inline const ModelConfig& lookup_model(const std::string& key) {
  auto& models = model_registry();
  auto it = models.find(key);
  if (it == models.end()) {
    fprintf(stderr, "Unknown model: '%s'. Options:", key.c_str());
    for (auto& [k, _] : models) fprintf(stderr, " %s", k.c_str());
    fprintf(stderr, "\n");
    exit(1);
  }
  return it->second;
}