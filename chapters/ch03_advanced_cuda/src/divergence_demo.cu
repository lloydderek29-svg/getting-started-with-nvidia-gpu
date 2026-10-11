#include <cstdio>
#include <cuda_runtime.h>
#include "gpubook/cuda_check.cuh"
#include "gpubook/timer.cuh"

__device__ float path_a(float v) {
  for (int k = 0; k < 200; ++k) v = sinf(v) + 1.0f;
  return v;
}

__device__ float path_b(float v) {
  for (int k = 0; k < 200; ++k) v = cosf(v) - 1.0f;
  return v;
}

__global__ void divergent(float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = (float)i;
  out[i] = (threadIdx.x % 2 == 0) ? path_a(v) : path_b(v);
}

__global__ void uniform(float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = (float)i;
  bool even_warp = ((threadIdx.x / 32) % 2 == 0);
  out[i] = even_warp ? path_a(v) : path_b(v);
}

int main() {
  const int n = 1 << 22;
  float* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
  const int block = 256;
  const int grid = (n + block - 1) / block;

  gpubook::GpuTimer timer;
  divergent<<<grid, block>>>(d, n);  // warm-up
  uniform<<<grid, block>>>(d, n);
  CUDA_CHECK(cudaDeviceSynchronize());

  timer.start();
  divergent<<<grid, block>>>(d, n);
  float t_div = timer.stop();

  timer.start();
  uniform<<<grid, block>>>(d, n);
  float t_uni = timer.stop();
  CUDA_CHECK(cudaGetLastError());

  std::printf("Divergent warps : %7.3f ms\n", t_div);
  std::printf("Uniform warps   : %7.3f ms\n", t_uni);
  std::printf("Slowdown        : %7.2fx\n", t_div / t_uni);

  CUDA_CHECK(cudaFree(d));
  return 0;
}
