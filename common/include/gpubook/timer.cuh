#pragma once
#include <cuda_runtime.h>
#include "gpubook/cuda_check.cuh"

namespace gpubook {

class GpuTimer {
 public:
  GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
  }
  ~GpuTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }
  GpuTimer(const GpuTimer&) = delete;
  GpuTimer& operator=(const GpuTimer&) = delete;

  void start(cudaStream_t s = 0) {
    CUDA_CHECK(cudaEventRecord(start_, s));
  }

  float stop(cudaStream_t s = 0) {
    CUDA_CHECK(cudaEventRecord(stop_, s));
    CUDA_CHECK(cudaEventSynchronize(stop_));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
  }

 private:
  cudaEvent_t start_{};
  cudaEvent_t stop_{};
};

}  // namespace gpubook
