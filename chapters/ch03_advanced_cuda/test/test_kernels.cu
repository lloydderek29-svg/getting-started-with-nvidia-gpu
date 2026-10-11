#include <algorithm>
#include <cmath>
#include <random>
#include <vector>
#include <gtest/gtest.h>
#include "ch03/kernels.hpp"
#include "gpubook/device_buffer.cuh"

namespace {

std::vector<float> random_vec(size_t n, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(gen);
  return v;
}

}  // namespace

class SumSizes : public ::testing::TestWithParam<size_t> {};

TEST_P(SumSizes, MatchesDoubleCpuReference) {
  const size_t n = GetParam();
  auto h = random_vec(n, 42);

  double ref = 0.0, abs_sum = 0.0;
  for (float v : h) {
    ref += v;
    abs_sum += std::fabs(v);
  }

  gpubook::DeviceBuffer<float> d(n);
  d.copy_from_host(h.data());
  float got = ch03::gpu_sum(d.data(), n);

  // Float summation error scales with the magnitude of the
  // terms, so the tolerance is relative to the sum of |x|.
  EXPECT_NEAR(got, ref, 1e-6 * abs_sum + 1e-6);
}

INSTANTIATE_TEST_SUITE_P(
    EdgeSizes, SumSizes,
    ::testing::Values(1, 31, 32, 33, 1023, 1024, 1025,
                      1000003));

struct Dims { int m, n, k; };
class MatmulDims : public ::testing::TestWithParam<Dims> {};

TEST_P(MatmulDims, MatchesCpuReference) {
  const Dims d = GetParam();
  auto a = random_vec((size_t)d.m * d.k, 1);
  auto b = random_vec((size_t)d.k * d.n, 2);
  std::vector<float> got((size_t)d.m * d.n);

  gpubook::DeviceBuffer<float> da(a.size()), db(b.size()),
      dc(got.size());
  da.copy_from_host(a.data());
  db.copy_from_host(b.data());
  ch03::gpu_matmul(da.data(), db.data(), dc.data(),
                   d.m, d.n, d.k);
  dc.copy_to_host(got.data());
  CUDA_CHECK_THROW(cudaDeviceSynchronize());

  for (int r = 0; r < d.m; ++r) {
    for (int c = 0; c < d.n; ++c) {
      double acc = 0.0;
      for (int t = 0; t < d.k; ++t) {
        acc += (double)a[(size_t)r * d.k + t] *
               b[(size_t)t * d.n + c];
      }
      ASSERT_NEAR(got[(size_t)r * d.n + c], acc,
                  1e-4 * d.k);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    TileEdges, MatmulDims,
    ::testing::Values(Dims{1, 1, 1}, Dims{17, 33, 65},
                      Dims{16, 16, 16}, Dims{128, 96, 80}));
