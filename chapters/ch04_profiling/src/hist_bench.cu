#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include "ch04/data.hpp"
#include "ch04/histogram.cuh"
#include "gpubook/bench.cuh"
#include "gpubook/cuda_check.cuh"

struct Variant {
  const char* name;
  void (*launch)(const uint8_t*, unsigned*, size_t,
                 cudaStream_t);
};

static void row(const char* kernel, const char* dist,
                const gpubook::Stats& s, size_t bytes) {
  std::printf("%s,%s,%.4f,%.4f,%.4f,%.1f\n", kernel, dist,
              s.median, s.p05, s.p95,
              bytes / (s.median * 1e-3) / 1e9);
}

int main() {
  const size_t n = 64ull << 20;
  const size_t bin_bytes = ch04::kBins * sizeof(unsigned);
  const Variant variants[] = {
      {"global_atomics", ch04::launch_hist_global},
      {"shared_private", ch04::launch_hist_shared},
      {"shared_vec16", ch04::launch_hist_shared_vec},
      {"warp_aggregated", ch04::launch_hist_warp_agg},
  };

  uint8_t* d_in = nullptr;
  unsigned* d_bins = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, n));
  CUDA_CHECK(cudaMalloc(&d_bins, bin_bytes));

  std::printf("kernel,dist,median_ms,p05_ms,p95_ms,gbps\n");
  for (int skew = 0; skew < 2; ++skew) {
    const char* dist = skew ? "skewed" : "uniform";
    auto host = skew ? ch04::make_skewed(n, 1)
                     : ch04::make_uniform(n, 1);
    auto ref = ch04::cpu_hist(host);
    CUDA_CHECK(cudaMemcpy(d_in, host.data(), n,
                          cudaMemcpyHostToDevice));

    gpubook::BenchConfig cpu_cfg{1, 5, false};
    auto cpu = gpubook::bench_cpu(
        [&] { ref = ch04::cpu_hist(host); }, cpu_cfg);
    row("cpu_1thread", dist, cpu, n);

    gpubook::BenchConfig cfg;
    cfg.flush_l2 = true;
    for (const auto& v : variants) {
      auto launch = [&](cudaStream_t s) {
        CUDA_CHECK(cudaMemsetAsync(d_bins, 0, bin_bytes, s));
        v.launch(d_in, d_bins, n, s);
      };
      auto stats = gpubook::bench_gpu(launch, cfg);

      std::vector<unsigned> got(ch04::kBins);
      CUDA_CHECK(cudaMemcpy(got.data(), d_bins, bin_bytes,
                            cudaMemcpyDeviceToHost));
      if (got != ref) {
        std::fprintf(stderr, "WRONG RESULT: %s (%s)\n",
                     v.name, dist);
        return 1;
      }
      row(v.name, dist, stats, n);
    }
  }
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_bins));
  return 0;
}
