#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <numeric>
#include <vector>
#include <cuda_runtime.h>
#include "gpubook/cuda_check.cuh"
#include "gpubook/timer.cuh"

namespace gpubook {

struct BenchConfig {
  int warmup = 5;
  int repeats = 30;
  bool flush_l2 = false;
};

struct Stats {
  double median = 0, mean = 0, stddev = 0;
  double p05 = 0, p95 = 0, min = 0, max = 0;
  int n = 0;
};

inline Stats summarize(std::vector<double> v) {
  Stats s;
  s.n = static_cast<int>(v.size());
  if (v.empty()) return s;
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  auto pick = [&](double q) {
    return v[static_cast<size_t>(q * (n - 1) + 0.5)];
  };
  s.median = (n % 2) ? v[n / 2]
                     : 0.5 * (v[n / 2 - 1] + v[n / 2]);
  s.mean = std::accumulate(v.begin(), v.end(), 0.0) / n;
  double var = 0.0;
  for (double x : v) var += (x - s.mean) * (x - s.mean);
  s.stddev = std::sqrt(var / std::max<size_t>(n - 1, 1));
  s.p05 = pick(0.05);
  s.p95 = pick(0.95);
  s.min = v.front();
  s.max = v.back();
  return s;
}

inline Stats bench_gpu(
    const std::function<void(cudaStream_t)>& fn,
    const BenchConfig& cfg = {},
    cudaStream_t stream = 0) {
  void* flush = nullptr;
  size_t flush_bytes = 0;
  if (cfg.flush_l2) {
    int dev = 0, l2 = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(
        &l2, cudaDevAttrL2CacheSize, dev));
    flush_bytes = static_cast<size_t>(l2) * 2;
    CUDA_CHECK(cudaMalloc(&flush, flush_bytes));
  }

  for (int i = 0; i < cfg.warmup; ++i) fn(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  GpuTimer timer;
  std::vector<double> ms;
  for (int i = 0; i < cfg.repeats; ++i) {
    if (flush) {
      CUDA_CHECK(cudaMemsetAsync(
          flush, i, flush_bytes, stream));
    }
    timer.start(stream);
    fn(stream);
    ms.push_back(timer.stop(stream));
  }
  CUDA_CHECK(cudaGetLastError());
  if (flush) CUDA_CHECK(cudaFree(flush));
  return summarize(ms);
}

inline Stats bench_cpu(const std::function<void()>& fn,
                       const BenchConfig& cfg = {}) {
  using clock = std::chrono::steady_clock;
  for (int i = 0; i < cfg.warmup; ++i) fn();
  std::vector<double> ms;
  for (int i = 0; i < cfg.repeats; ++i) {
    auto t0 = clock::now();
    fn();
    auto t1 = clock::now();
    ms.push_back(
        std::chrono::duration<double, std::milli>(t1 - t0)
            .count());
  }
  return summarize(ms);
}

}  // namespace gpubook
