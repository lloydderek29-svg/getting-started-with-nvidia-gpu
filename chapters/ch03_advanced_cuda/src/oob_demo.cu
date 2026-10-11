#include <cstdio>
#include <cuda_runtime.h>
#include "gpubook/cuda_check.cuh"

__global__ void fill(int* data, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i <= n) {          // BUG: should be i < n
    data[i] = i;
  }
}

int main() {
  const int n = 1000;
  int* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, n * sizeof(int)));
  fill<<<(n + 255) / 256, 256>>>(d, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::printf("Finished without a visible error\n");
  CUDA_CHECK(cudaFree(d));
  return 0;
}
