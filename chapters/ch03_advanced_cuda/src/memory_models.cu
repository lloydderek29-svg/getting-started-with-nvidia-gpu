#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "gpubook/cuda_check.cuh"
#include "gpubook/managed.cuh"
#include "gpubook/timer.cuh"

__global__ void scale(const float* in, float* out,
                      size_t n, float a) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) out[i] = in[i] * a;
}

static bool verify(const float* out, size_t n, float a) {
  for (size_t i = 0; i < n; ++i) {
    if (out[i] != a * (float)(i % 1000)) return false;
  }
  return true;
}

static void fill(float* p, size_t n) {
  for (size_t i = 0; i < n; ++i) p[i] = (float)(i % 1000);
}

static float run_explicit(float* h_in, float* h_out,
                          size_t n) {
  const size_t bytes = n * sizeof(float);
  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  gpubook::GpuTimer t;
  t.start();
  CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes,
                        cudaMemcpyHostToDevice));
  scale<<<1024, 256>>>(d_in, d_out, n, 2.0f);
  CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes,
                        cudaMemcpyDeviceToHost));
  float ms = t.stop();
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return ms;
}

static float run_managed(size_t n, int dev) {
  const size_t bytes = n * sizeof(float);
  float *in, *out;
  CUDA_CHECK(cudaMallocManaged(&in, bytes));
  CUDA_CHECK(cudaMallocManaged(&out, bytes));
  fill(in, n);  // first touch happens on the CPU

  gpubook::GpuTimer t;
  t.start();
  CUDA_CHECK(gpubook::prefetch_to_device(in, bytes, dev));
  CUDA_CHECK(gpubook::prefetch_to_device(out, bytes, dev));
  scale<<<1024, 256>>>(in, out, n, 2.0f);
  CUDA_CHECK(cudaDeviceSynchronize());
  float ms = t.stop();

  bool ok = verify(out, n, 2.0f);  // migrates back to CPU
  std::printf("  managed result check: %s\n",
              ok ? "OK" : "FAILED");
  CUDA_CHECK(cudaFree(in));
  CUDA_CHECK(cudaFree(out));
  return ms;
}

int main() {
  const size_t n = 1ull << 26;  // 256 MiB per array
  const size_t bytes = n * sizeof(float);
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));

  float* pageable_in = (float*)std::malloc(bytes);
  float* pageable_out = (float*)std::malloc(bytes);
  fill(pageable_in, n);
  run_explicit(pageable_in, pageable_out, n);  // warm-up
  float t_pageable =
      run_explicit(pageable_in, pageable_out, n);
  std::printf("Pageable + explicit : %8.2f ms (%s)\n",
              t_pageable,
              verify(pageable_out, n, 2.0f) ? "OK" : "BAD");

  float *pin_in, *pin_out;
  CUDA_CHECK(cudaMallocHost(&pin_in, bytes));
  CUDA_CHECK(cudaMallocHost(&pin_out, bytes));
  fill(pin_in, n);
  float t_pinned = run_explicit(pin_in, pin_out, n);
  std::printf("Pinned   + explicit : %8.2f ms (%s)\n",
              t_pinned,
              verify(pin_out, n, 2.0f) ? "OK" : "BAD");

  float t_managed = run_managed(n, dev);
  std::printf("Managed  + prefetch : %8.2f ms\n", t_managed);

  CUDA_CHECK(cudaFreeHost(pin_in));
  CUDA_CHECK(cudaFreeHost(pin_out));
  std::free(pageable_in);
  std::free(pageable_out);
  return 0;
}
