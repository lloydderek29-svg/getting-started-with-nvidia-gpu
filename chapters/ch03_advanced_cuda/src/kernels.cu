float ch03::gpu_sum(const float* d_in, size_t n) {
  float* d_out = nullptr;
  CUDA_CHECK_THROW(cudaMalloc(&d_out, sizeof(float)));
  CUDA_CHECK_THROW(cudaMemset(d_out, 0, sizeof(float)));

  const int block = 256;
  const int grid = 128;
  reduce_sum<<<grid, block>>>(d_in, d_out, n);
  CUDA_CHECK_THROW(cudaGetLastError());

  float result = 0.0f;
  CUDA_CHECK_THROW(cudaMemcpy(&result, d_out, sizeof(float),
                              cudaMemcpyDeviceToHost));
  CUDA_CHECK_THROW(cudaFree(d_out));
  return result;
}
