// chapters/ch03_advanced_cuda/src/matrix_2d.cu
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include "gpubook/cuda_check.cuh"

__global__ void add_2d(const float* a, const float* b,
                       float* c, int rows, int cols) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < rows && col < cols) {
    int idx = row * cols + col;
    c[idx] = a[idx] + b[idx];
  }
}

int main() {
  const int rows = 1000, cols = 1500;
  const size_t n = (size_t)rows * cols;
  std::vector<float> ha(n, 1.5f), hb(n, 2.5f), hc(n);

  float *a, *b, *c;
  CUDA_CHECK(cudaMalloc(&a, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&b, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&c, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(a, ha.data(), n * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b, hb.data(), n * sizeof(float),
                        cudaMemcpyHostToDevice));

  dim3 block(16, 16);
  dim3 grid((cols + block.x - 1) / block.x,
            (rows + block.y - 1) / block.y);
  add_2d<<<grid, block>>>(a, b, c, rows, cols);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(hc.data(), c, n * sizeof(float),
                        cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < n; ++i) {
    if (hc[i] != 4.0f) {
      std::printf("Mismatch at %zu\n", i);
      return 1;
    }
  }
  std::printf("2D add: OK (%d x %d)\n", rows, cols);

  CUDA_CHECK(cudaFree(a));
  CUDA_CHECK(cudaFree(b));
  CUDA_CHECK(cudaFree(c));
  return 0;
}
