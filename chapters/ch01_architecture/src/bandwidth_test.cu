#include <algorithm>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include "gpubook/cuda_check.cuh"
#include "gpubook/timer.cuh"

__global__ void copy_kernel(const float4* __restrict__ in,
                            float4* __restrict__ out,
                            size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) {
    out[i] = in[i];
  }
}

static int get_attr(cudaDeviceAttr a, int dev) {
  int v = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&v, a, dev));
  return v;
}

int main() {
  const size_t bytes = 512ull << 20;
  const size_t n = bytes / sizeof(float4);

  float4* d_in = nullptr;
  float4* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  CUDA_CHECK(cudaMemset(d_in, 1, bytes));

  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  int sms = get_attr(cudaDevAttrMultiProcessorCount, dev);
  int bus = get_attr(cudaDevAttrGlobalMemoryBusWidth, dev);
  int mclk = get_attr(cudaDevAttrMemoryClockRate, dev);

  const int block = 256;
  const int grid = sms * 8;

  for (int i = 0; i < 3; ++i) {
    copy_kernel<<<grid, block>>>(d_in, d_out, n);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  gpubook::GpuTimer timer;
  std::vector<float> times;
  for (int i = 0; i < 20; ++i) {
    timer.start();
    copy_kernel<<<grid, block>>>(d_in, d_out, n);
    times.push_back(timer.stop());
  }
  CUDA_CHECK(cudaGetLastError());

  std::sort(times.begin(), times.end());
  double ms = times[times.size() / 2];
  double measured = 2.0 * bytes / (ms * 1e-3) / 1e9;
  double peak = 2.0 * mclk * 1e3 * (bus / 8.0) / 1e9;

  std::printf("Measured copy bandwidth : %7.1f GB/s\n",
              measured);
  std::printf("Computed peak (2*clk*bus): %7.1f GB/s\n",
              peak);
  std::printf("Efficiency               : %7.1f %%\n",
              100.0 * measured / peak);

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return 0;
}
