#pragma once
#include <cstddef>
#include <cuda_runtime.h>
#include "gpubook/cuda_check.cuh"

namespace gpubook {

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t n) : n_(n) {
    CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(T)));
  }
  ~DeviceBuffer() {
    if (ptr_) cudaFree(ptr_);
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& o) noexcept
      : ptr_(o.ptr_), n_(o.n_) {
    o.ptr_ = nullptr;
    o.n_ = 0;
  }
  DeviceBuffer& operator=(DeviceBuffer&&) = delete;

  T* data() { return ptr_; }
  const T* data() const { return ptr_; }
  size_t size() const { return n_; }
  size_t bytes() const { return n_ * sizeof(T); }

  void copy_from_host(const T* src, cudaStream_t s = 0) {
    CUDA_CHECK(cudaMemcpyAsync(ptr_, src, bytes(),
                               cudaMemcpyHostToDevice, s));
  }
  void copy_to_host(T* dst, cudaStream_t s = 0) const {
    CUDA_CHECK(cudaMemcpyAsync(dst, ptr_, bytes(),
                               cudaMemcpyDeviceToHost, s));
  }

 private:
  T* ptr_ = nullptr;
  size_t n_ = 0;
};

}  // namespace gpubook
