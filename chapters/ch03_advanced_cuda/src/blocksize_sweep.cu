#include <algorithm>
#include <cstdio>
#include <cuda_runtime.h>
#include "gpubook/cuda_check.cuh"
#include "gpubook/timer.cuh"

__global__ void saxpy(int n, float a,
                      const float* x, float* y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; i < n; i += stride) {
    y[i] = a * x[i] + y[i];
  }
}

int main() {
  const int n = 1 << 25;
  const size_t bytes = n * sizeof(float);

  float *x = nullptr, *y = nullptr;
  CUDA_CHECK(cudaMalloc(&x, bytes));
  CUDA_CHECK(cudaMalloc(&y, bytes));
  CUDA_CHECK(cudaMemset(x, 0, bytes));
  CUDA_CHECK(cudaMemset(y, 0, bytes));

  int dev = 0, sms = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaDeviceGetAttribute(
      &sms, cudaDevAttrMultiProcessorCount, dev));

  std::printf("%8s %8s %10s %10s\n",
              "block", "grid", "ms", "GB/s");
  const int sizes[] = {32, 64, 128, 256, 512, 1024};
  gpubook::GpuTimer timer;

  for (int block : sizes) {
    int grid = std::min((n + block - 1) / block, sms * 16);
    saxpy<<<grid, block>>>(n, 2.0f, x, y);  // warm-up
    CUDA_CHECK(cudaDeviceSynchronize());

    const int reps = 20;
    timer.start();
    for (int r = 0; r < reps; ++r) {
      saxpy<<<grid, block>>>(n, 2.0f, x, y);
    }
    float ms = timer.stop() / reps;
    CUDA_CHECK(cudaGetLastError());

    double moved = 3.0 * bytes;  // read x, read y, write y
    std::printf("%8d %8d %10.3f %10.1f\n", block, grid,
                ms, moved / (ms * 1e-3) / 1e9);
  }

  CUDA_CHECK(cudaFree(x));
  CUDA_CHECK(cudaFree(y));
  return 0;
}
