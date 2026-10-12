#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include "ch04/data.hpp"
#include "ch04/histogram.cuh"
#include "gpubook/bench.cuh"
#include "gpubook/cuda_check.cuh"

static void report(const char* label,
                   const gpubook::Stats& s, size_t bytes) {
  double gbs = bytes / (s.median * 1e-3) / 1e9;
  std::printf("%-22s %8.3f ms  [p05 %7.3f, p95 %7.3f]"
              "  %7.1f GB/s\n",
              label, s.median, s.p05, s.p95, gbs);
}

int main() {
  const size_t n = 64ull << 20;  // 64 MiB of pixels
  auto host = ch04::make_uniform(n, 1);
  auto ref = ch04::cpu_hist(host);

  gpubook::BenchConfig cpu_cfg{2, 10, false};
  std::vector<unsigned> cpu_bins;
  auto cpu = gpubook::bench_cpu(
      [&] { cpu_bins = ch04::cpu_hist(host); }, cpu_cfg);
  report("CPU, 1 thread", cpu, n);

  uint8_t* d_in = nullptr;
  unsigned* d_bins = nullptr;
  const size_t bin_bytes = ch04::kBins * sizeof(unsigned);
  CUDA_CHECK(cudaMalloc(&d_in, n));
  CUDA_CHECK(cudaMalloc(&d_bins, bin_bytes));
  CUDA_CHECK(cudaMemcpy(d_in, host.data(), n,
                        cudaMemcpyHostToDevice));

  gpubook::BenchConfig cfg;
  cfg.flush_l2 = true;

  auto kernel_only = gpubook::bench_gpu(
      [&](cudaStream_t s) {
        CUDA_CHECK(cudaMemsetAsync(d_bins, 0, bin_bytes, s));
        ch04::launch_hist_global(d_in, d_bins, n, s);
      },
      cfg);
  report("GPU kernel only", kernel_only, n);

  std::vector<unsigned> gpu_bins(ch04::kBins);
  CUDA_CHECK(cudaMemcpy(gpu_bins.data(), d_bins, bin_bytes,
                        cudaMemcpyDeviceToHost));
  std::printf("Result matches CPU: %s\n",
              gpu_bins == ref ? "yes" : "NO");

  auto end_to_end = gpubook::bench_gpu(
      [&](cudaStream_t s) {
        CUDA_CHECK(cudaMemcpy(d_in, host.data(), n,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemsetAsync(d_bins, 0, bin_bytes, s));
        ch04::launch_hist_global(d_in, d_bins, n, s);
        CUDA_CHECK(cudaMemcpy(gpu_bins.data(), d_bins,
                              bin_bytes,
                              cudaMemcpyDeviceToHost));
      });
  report("GPU end to end", end_to_end, n);

  std::printf("\nSpeedup vs CPU, kernel only : %.1fx\n",
              cpu.median / kernel_only.median);
  std::printf("Speedup vs CPU, end to end  : %.1fx\n",
              cpu.median / end_to_end.median);

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_bins));
  return 0;
}
