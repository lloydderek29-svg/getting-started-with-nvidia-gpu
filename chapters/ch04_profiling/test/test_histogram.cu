#include <vector>
#include <gtest/gtest.h>
#include "ch04/data.hpp"
#include "ch04/histogram.cuh"
#include "gpubook/device_buffer.cuh"

namespace {

using Launch = void (*)(const uint8_t*, unsigned*, size_t,
                        cudaStream_t);

struct Variant {
  const char* name;
  Launch launch;
};

std::vector<unsigned> run(Launch launch,
                          const std::vector<uint8_t>& h) {
  gpubook::DeviceBuffer<uint8_t> in(h.size());
  gpubook::DeviceBuffer<unsigned> bins(ch04::kBins);
  in.copy_from_host(h.data());
  CUDA_CHECK_THROW(cudaMemset(
      bins.data(), 0, ch04::kBins * sizeof(unsigned)));
  launch(in.data(), bins.data(), h.size(), 0);
  CUDA_CHECK_THROW(cudaGetLastError());
  std::vector<unsigned> out(ch04::kBins);
  bins.copy_to_host(out.data());
  CUDA_CHECK_THROW(cudaDeviceSynchronize());
  return out;
}

}  // namespace

TEST(Histogram, AllVariantsMatchCpuReference) {
  const Variant variants[] = {
      {"global", ch04::launch_hist_global},
      {"shared", ch04::launch_hist_shared},
      {"shared_vec", ch04::launch_hist_shared_vec},
      {"warp_agg", ch04::launch_hist_warp_agg},
  };
  const size_t sizes[] = {1,  15,   16,   17,
                          255, 4097, (1u << 20) + 13};
  for (const auto& v : variants) {
    for (size_t n : sizes) {
      for (int skew = 0; skew < 2; ++skew) {
        SCOPED_TRACE(std::string(v.name) + " n=" +
                     std::to_string(n) +
                     (skew ? " skewed" : " uniform"));
        auto h = skew ? ch04::make_skewed(n, 3)
                      : ch04::make_uniform(n, 3);
        EXPECT_EQ(run(v.launch, h), ch04::cpu_hist(h));
      }
    }
  }
}
