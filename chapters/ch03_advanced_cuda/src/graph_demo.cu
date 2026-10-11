#include <cstdio>
#include <cuda_runtime.h>
#include "gpubook/cuda_check.cuh"
#include "gpubook/timer.cuh"

__global__ void step(float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = x[i] * 1.0001f + 0.0001f;
}

int main() {
  const int n = 1 << 16;
  const int kernels_per_iter = 20;
  const int iters = 1000;
  const int block = 256;
  const int grid = (n + block - 1) / block;

  float* d_x;
  CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_x, 0, n * sizeof(float)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  gpubook::GpuTimer timer;

  // Baseline: launch every kernel individually.
  timer.start(stream);
  for (int it = 0; it < iters; ++it) {
    for (int k = 0; k < kernels_per_iter; ++k) {
      step<<<grid, block, 0, stream>>>(d_x, n);
    }
  }
  float t_plain = timer.stop(stream);

  // Capture one iteration into a graph.
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  CUDA_CHECK(cudaStreamBeginCapture(
      stream, cudaStreamCaptureModeGlobal));
  for (int k = 0; k < kernels_per_iter; ++k) {
    step<<<grid, block, 0, stream>>>(d_x, n);
  }
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));

  // Replay it.
  CUDA_CHECK(cudaGraphLaunch(exec, stream));  // warm-up
  CUDA_CHECK(cudaStreamSynchronize(stream));
  timer.start(stream);
  for (int it = 0; it < iters; ++it) {
    CUDA_CHECK(cudaGraphLaunch(exec, stream));
  }
  float t_graph = timer.stop(stream);

  std::printf("Individual launches: %8.2f ms\n", t_plain);
  std::printf("Graph replay       : %8.2f ms\n", t_graph);
  std::printf("Speedup            : %8.2fx\n",
              t_plain / t_graph);

  CUDA_CHECK(cudaGraphExecDestroy(exec));
  CUDA_CHECK(cudaGraph_destroy_placeholder(graph));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
