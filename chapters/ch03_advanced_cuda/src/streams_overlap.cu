#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>
#include "gpubook/cuda_check.cuh"

__global__ void heavy(const float* in, float* out, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) {
    float v = in[i];
    for (int k = 0; k < 64; ++k) {
      v = sinf(v) * 0.5f + v * 0.5f;
    }
    out[i] = v;
  }
}

static double wall_ms() {
  using clock = std::chrono::steady_clock;
  auto t = clock::now().time_since_epoch();
  return std::chrono::duration<double, std::milli>(t).count();
}

int main() {
  const int kStreams = 4;
  const size_t n = 1 << 26;
  const size_t bytes = n * sizeof(float);
  const size_t chunk = n / kStreams;
  const size_t cbytes = chunk * sizeof(float);

  float *h_in, *h_out;
  CUDA_CHECK(cudaMallocHost(&h_in, bytes));
  CUDA_CHECK(cudaMallocHost(&h_out, bytes));
  for (size_t i = 0; i < n; ++i) h_in[i] = (float)(i % 100);

  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));

  const int block = 256, grid = 1024;

  // Serial baseline: one stream, whole array.
  heavy<<<grid, block>>>(d_in, d_out, n);  // warm-up
  CUDA_CHECK(cudaDeviceSynchronize());
  double t0 = wall_ms();
  CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes,
                        cudaMemcpyHostToDevice));
  heavy<<<grid, block>>>(d_in, d_out, n);
  CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());
  double serial = wall_ms() - t0;

  // Pipelined: chunks spread over several streams.
  cudaStream_t streams[kStreams];
  for (int s = 0; s < kStreams; ++s) {
    CUDA_CHECK(cudaStreamCreateWithFlags(
        &streams[s], cudaStreamNonBlocking));
  }

  t0 = wall_ms();
  for (int s = 0; s < kStreams; ++s) {
    size_t off = s * chunk;
    CUDA_CHECK(cudaMemcpyAsync(d_in + off, h_in + off,
        cbytes, cudaMemcpyHostToDevice, streams[s]));
    heavy<<<grid / kStreams, block, 0, streams[s]>>>(
        d_in + off, d_out + off, chunk);
    CUDA_CHECK(cudaMemcpyAsync(h_out + off, d_out + off,
        cbytes, cudaMemcpyDeviceToHost, streams[s]));
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  double overlapped = wall_ms() - t0;

  std::printf("Serial     : %8.2f ms\n", serial);
  std::printf("%d streams  : %8.2f ms\n", kStreams, overlapped);
  std::printf("Speedup    : %8.2fx\n", serial / overlapped);

  for (int s = 0; s < kStreams; ++s) {
    CUDA_CHECK(cudaStreamDestroy(streams[s]));
  }
  CUDA_CHECK(cudaFreeHost(h_in));
  CUDA_CHECK(cudaFreeHost(h_out));
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return 0;
}
