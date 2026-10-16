# Chapter 3: Advanced CUDA C++ Programming and Memory Management

This directory contains the CUDA C++ demos for execution models, memory management, shared memory kernels, streams, and graphs, together with a GoogleTest suite and a sanitizer demo. Run every command from the repository root.

## What Is Here

| Path | Topic |
|---|---|
| `src/blocksize_sweep.cu` | Effect of block size on a memory-bound kernel |
| `src/matrix_2d.cu` | Two-dimensional grids and bounds checks |
| `src/divergence_demo.cu` | Cost of warp divergence |
| `src/occupancy_demo.cu` | Occupancy API |
| `src/memory_models.cu` | Pageable, pinned, and managed memory compared |
| `src/reduce_sum.cu` | Warp shuffle reduction against CUB |
| `src/matmul_tiled.cu` | Naive and tiled matrix multiplication |
| `src/transpose.cu` | Padded shared memory transpose |
| `src/streams_overlap.cu` | Overlapping copies and compute |
| `src/graph_demo.cu` | CUDA Graph replay |
| `src/oob_demo.cu` | A deliberate out-of-bounds bug for the sanitizer |
| `src/kernels.cu` | Kernels and wrappers used by the tests |
| `include/ch03/kernels.hpp` | Public interface of the tested kernels |
| `tests/test_kernels.cu` | Reference-checked tests at awkward sizes |

Shared headers: `gpubook/cuda_check.cuh`, `timer.cuh`, `device_buffer.cuh`, and `managed.cuh`.

## Build

Make sure the `DEMOS` list in `CMakeLists.txt` names every demo.

```cmake
set(DEMOS
  blocksize_sweep matrix_2d occupancy_demo divergence_demo
  memory_models reduce_sum matmul_tiled transpose
  streams_overlap graph_demo oob_demo
)
```

```bash
cmake --preset release
cmake --build --preset release -j
```

The test target downloads GoogleTest during configuration, so the first configure needs network access.

## Run and What to Expect

```bash
./build/release/bin/blocksize_sweep
```

Block sizes of 128 and above should reach nearly the same bandwidth, with 32 clearly slower.

```bash
./build/release/bin/matrix_2d
./build/release/bin/divergence_demo
./build/release/bin/occupancy_demo
```

The divergence demo should report a slowdown near 2x for the divergent kernel. Occupancy for the small kernel should be at or near 100 percent.

```bash
./build/release/bin/memory_models
```

Pinned memory should beat pageable, and managed memory with prefetch should land close to pinned. Remove the prefetch calls to see the cost of page-fault migration.

```bash
./build/release/bin/reduce_sum
./build/release/bin/matmul_tiled
./build/release/bin/transpose
```

The reduction prints both sums, and they should agree to within float rounding. The tiled kernel should run several times faster than the naive one, though far below cuBLAS. The transpose prints `OK`.

```bash
./build/release/bin/streams_overlap
./build/release/bin/graph_demo
```

The streams demo reports a speedup above 1x when copies and compute are balanced. The graph demo gives its largest gain on tiny kernels and should show a clear speedup. In `graph_demo.cu`, release the graph with `cudaGraphDestroy(graph)`.

## Tests and Sanitizers

```bash
ctest --preset release --output-on-failure
```

The tests compare the reduction and matrix kernels against double-precision CPU references at sizes such as 1, 31, 32, 33, 1023, 1024, 1025, and 1000003. The `ch03_memcheck` test runs the suite under `compute-sanitizer` and needs the toolkit's sanitizer on `PATH`.

Run the sanitizer manually on the buggy demo.

```bash
compute-sanitizer --tool memcheck ./build/release/bin/oob_demo
```

```text
========= Invalid __global__ write of size 4 bytes
=========     at fill(int *, int)+0x70
=========     by thread (232,0,0) in block (3,0,0)
```

The other tools are `racecheck`, `initcheck`, and `synccheck`. Use small inputs, because sanitizer runs are slow.

## Debugging

```bash
cmake --preset debug
cmake --build --preset debug -j
cuda-gdb ./build/debug/bin/oob_demo
```

To attribute asynchronous errors to the launch that caused them, configure with `-DCMAKE_CUDA_FLAGS="-DGPUBOOK_DEBUG_SYNC"`, or set `CUDA_LAUNCH_BLOCKING=1` for one run. Remove either before benchmarking.

## Troubleshooting

| Problem | Fix |
|---|---|
| Configure fails fetching GoogleTest | Allow network access or vendor the archive |
| `cub/cub.cuh` not found | Use the toolkit's own headers and check `nvcc --version` |
| Results differ slightly between runs | Expected for atomic float sums, so tests use tolerances |
| Sanitizer reports nothing on `oob_demo` | Build with `-lineinfo` and run the memcheck tool |