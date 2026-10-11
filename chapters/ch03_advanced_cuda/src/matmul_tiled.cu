#include <cstdio>
#include <cuda_runtime.h>
#include "gpubook/cuda_check.cuh"
#include "gpubook/timer.cuh"

#define TILE 16

__global__ void matmul_naive(const float* A, const float* B,
                             float* C, int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
  }
}

__global__ void matmul_tiled(const float* A, const float* B,
                             float* C, int M, int N, int K) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;
  float acc = 0.0f;

  int tiles = (K + TILE - 1) / TILE;
  for (int t = 0; t < tiles; ++t) {
    int a_col = t * TILE + threadIdx.x;
    int b_row = t * TILE + threadIdx.y;

    As[threadIdx.y][threadIdx.x] =
        (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] =
        (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
    __syncthreads();

    for (int k = 0; k < TILE; ++k) {
      acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < M && col < N) C[row * N + col] = acc;
}

int main() {
  const int M = 2048, N = 2048, K = 2048;
  const size_t sa = (size_t)M * K, sb = (size_t)K * N,
               sc = (size_t)M * N;

  float *A, *B, *C;
  CUDA_CHECK(cudaMalloc(&A, sa * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&B, sb * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&C, sc * sizeof(float)));
  CUDA_CHECK(cudaMemset(A, 0, sa * sizeof(float)));
  CUDA_CHECK(cudaMemset(B, 0, sb * sizeof(float)));

  dim3 block(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  const double flops = 2.0 * M * N * K;
  gpubook::GpuTimer timer;

  matmul_naive<<<grid, block>>>(A, B, C, M, N, K);
  matmul_tiled<<<grid, block>>>(A, B, C, M, N, K);
  CUDA_CHECK(cudaDeviceSynchronize());

  timer.start();
  matmul_naive<<<grid, block>>>(A, B, C, M, N, K);
  float t_naive = timer.stop();

  timer.start();
  matmul_tiled<<<grid, block>>>(A, B, C, M, N, K);
  float t_tiled = timer.stop();
  CUDA_CHECK(cudaGetLastError());

  std::printf("Naive : %8.3f ms  %7.1f GFLOPS\n", t_naive,
              flops / (t_naive * 1e-3) / 1e9);
  std::printf("Tiled : %8.3f ms  %7.1f GFLOPS\n", t_tiled,
              flops / (t_tiled * 1e-3) / 1e9);

  CUDA_CHECK(cudaFree(A));
  CUDA_CHECK(cudaFree(B));
  CUDA_CHECK(cudaFree(C));
  return 0;
}
