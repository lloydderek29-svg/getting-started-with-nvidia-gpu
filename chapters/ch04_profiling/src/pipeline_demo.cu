#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>
#include "ch04/data.hpp"
#include "ch04/histogram.cuh"
#include "gpubook/cuda_check.cuh"

namespace {

constexpr size_t kBatchBytes = 64ull << 20;
constexpr size_t kBinBytes = ch04::kBins * sizeof(unsigned);
constexpr int kBatches = 16;
constexpr int kSlots = 2;

double now_ms() {
  using clock = std::chrono::steady_clock;
  auto t = clock::now().time_since_epoch();
  return std::chrono::duration<double, std::milli>(t).count();
}

// Stand-in for decoding: copy a template, stamp the batch id.
void load_batch(uint8_t* dst, const uint8_t* tmpl, int id) {
  nvtx3::scoped_range r{"load_batch"};
  std::memcpy(dst, tmpl, kBatchBytes);
  dst[0] = static_cast<uint8_t>(id);
}

void accumulate(std::vector<unsigned long long>& total,
                const unsigned* part) {
  for (int b = 0; b < ch04::kBins; ++b) total[b] += part[b];
}

double run_serial(const uint8_t* tmpl,
                  std::vector<unsigned long long>& total) {
  std::vector<uint8_t> h_in(kBatchBytes);
  std::vector<unsigned> h_bins(ch04::kBins);
  uint8_t* d_in = nullptr;
  unsigned* d_bins = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, kBatchBytes));
  CUDA_CHECK(cudaMalloc(&d_bins, kBinBytes));

  double t0 = now_ms();
  for (int b = 0; b < kBatches; ++b) {
    nvtx3::scoped_range batch{"batch"};
    load_batch(h_in.data(), tmpl, b);
    {
      nvtx3::scoped_range r{"copy_in"};
      CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), kBatchBytes,
                            cudaMemcpyHostToDevice));
    }
    {
      nvtx3::scoped_range r{"kernel"};
      CUDA_CHECK(cudaMemset(d_bins, 0, kBinBytes));
      ch04::launch_hist_global(d_in, d_bins, kBatchBytes);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    {
      nvtx3::scoped_range r{"copy_out"};
      CUDA_CHECK(cudaMemcpy(h_bins.data(), d_bins, kBinBytes,
                            cudaMemcpyDeviceToHost));
    }
    accumulate(total, h_bins.data());
  }
  double ms = now_ms() - t0;
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_bins));
  return ms;
}

double run_overlap(const uint8_t* tmpl,
                   std::vector<unsigned long long>& total) {
  uint8_t *h_in[kSlots], *d_in[kSlots];
  unsigned *h_bins[kSlots], *d_bins[kSlots];
  cudaStream_t st[kSlots];
  for (int s = 0; s < kSlots; ++s) {
    CUDA_CHECK(cudaMallocHost(&h_in[s], kBatchBytes));
    CUDA_CHECK(cudaMalloc(&d_in[s], kBatchBytes));
    CUDA_CHECK(cudaMallocHost(&h_bins[s], kBinBytes));
    CUDA_CHECK(cudaMalloc(&d_bins[s], kBinBytes));
    CUDA_CHECK(cudaStreamCreateWithFlags(
        &st[s], cudaStreamNonBlocking));
  }

  double t0 = now_ms();
  for (int b = 0; b < kBatches; ++b) {
    const int s = b % kSlots;
    {
      nvtx3::scoped_range r{"wait_slot"};
      CUDA_CHECK(cudaStreamSynchronize(st[s]));
      if (b >= kSlots) accumulate(total, h_bins[s]);
    }
    load_batch(h_in[s], tmpl, b);
    CUDA_CHECK(cudaMemcpyAsync(d_in[s], h_in[s], kBatchBytes,
                               cudaMemcpyHostToDevice, st[s]));
    CUDA_CHECK(cudaMemsetAsync(d_bins[s], 0, kBinBytes,
                               st[s]));
    ch04::launch_hist_global(d_in[s], d_bins[s],
                             kBatchBytes, st[s]);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(h_bins[s], d_bins[s], kBinBytes,
                               cudaMemcpyDeviceToHost, st[s]));
  }
  for (int s = 0; s < kSlots; ++s) {
    CUDA_CHECK(cudaStreamSynchronize(st[s]));
    accumulate(total, h_bins[s]);
  }
  double ms = now_ms() - t0;

  for (int s = 0; s < kSlots; ++s) {
    CUDA_CHECK(cudaStreamDestroy(st[s]));
    CUDA_CHECK(cudaFreeHost(h_in[s]));
    CUDA_CHECK(cudaFree(d_in[s]));
    CUDA_CHECK(cudaFreeHost(h_bins[s]));
    CUDA_CHECK(cudaFree(d_bins[s]));
  }
  return ms;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "serial";
  auto tmpl = ch04::make_uniform(kBatchBytes, 7);

  {
    nvtx3::scoped_range r{"warmup"};
    CUDA_CHECK(cudaFree(nullptr));  // create the context
  }

  std::vector<unsigned long long> total(ch04::kBins, 0);
  double ms = (mode == "overlap")
                  ? run_overlap(tmpl.data(), total)
                  : run_serial(tmpl.data(), total);

  unsigned long long sum = 0;
  for (auto v : total) sum += v;
  const bool ok = sum == (unsigned long long)kBatches *
                             kBatchBytes;
  std::printf("%s: %.1f ms for %d batches (%s)\n",
              mode.c_str(), ms, kBatches,
              ok ? "counts verified" : "COUNT MISMATCH");
  return ok ? 0 : 1;
}
