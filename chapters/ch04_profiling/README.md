# Chapter 4: GPU Profiling and Performance Optimization

This directory contains benchmark harnesses, telemetry tools, profiler workflows, and an optimization case study: a 256-bin histogram taken from a naive kernel to a tuned one in measured steps. Run every command from the repository root.

## What Is Here

| Path | Purpose |
|---|---|
| `include/ch04/histogram.cuh` | Four histogram kernels and launch helpers |
| `include/ch04/data.hpp` | Uniform and skewed data generators and the CPU reference |
| `src/baseline_bench.cu` | CPU and GPU baselines |
| `src/pipeline_demo.cu` | Serial and overlapped pipeline with NVTX ranges |
| `src/hist_bench.cu` | Benchmark of all kernels on both distributions |
| `python/baseline_hist.py` | NumPy and PyTorch baselines |
| `python/monitor_demo.py` | Telemetry signatures of different workloads |
| `python/torch_nvtx_demo.py` | Annotated PyTorch loop for Nsight Systems |
| `python/ncu_summary.py` | Compact Nsight Compute table |
| `python/perf_report.py` | Converts benchmark CSV to a Markdown table |
| `tests/` | Harness, monitor, and histogram tests |

Shared code: `gpubook/bench.cuh`, `gpubook/bench.py`, and `gpubook/monitor.py`.

## Prerequisites

The base setup from Chapter 2, plus Nsight Systems (`nsys`) and Nsight Compute (`ncu`) from the CUDA Toolkit. The NVTX header `nvtx3/nvtx3.hpp` also ships with the toolkit.

## Build

```bash
cmake --preset release
cmake --build --preset release -j
mkdir -p results/ch04
```

All targets compile with `-lineinfo` so profilers can map results to source lines.

## Run

Baselines, which print kernel-only and end-to-end speedups:

```bash
./build/release/bin/baseline_bench
python chapters/ch04_profiling/python/baseline_hist.py
```

Telemetry demo, which runs four phases of about 15 seconds each:

```bash
python chapters/ch04_profiling/python/monitor_demo.py
```

Expect low power during idle, the highest power in the compute phase, and PCIe receive throughput near your pinned transfer rate in the last phase. A CSV is written to `results/ch04/monitor.csv`.

Timeline profiles:

```bash
nsys profile --trace=cuda,nvtx,osrt \
  --output=results/ch04/serial --force-overwrite=true \
  ./build/release/bin/pipeline_demo serial

nsys profile --trace=cuda,nvtx,osrt \
  --output=results/ch04/overlap --force-overwrite=true \
  ./build/release/bin/pipeline_demo overlap
```

Open the `.nsys-rep` files in Nsight Systems. The serial run shows gaps in the GPU rows while the CPU loads data, and the overlapped run is denser. Text summaries:

```bash
nsys stats --report cuda_gpu_kern_sum results/ch04/serial.nsys-rep
```

The PyTorch version, with and without the synchronization mistake:

```bash
nsys profile --trace=cuda,nvtx,cudnn,cublas \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --output=results/ch04/torch_loop --force-overwrite=true \
  python chapters/ch04_profiling/python/torch_nvtx_demo.py
```

Add `--sync-every-step` to the Python command to see the gaps.

Kernel analysis:

```bash
ncu --set full --kernel-name hist_global --launch-count 1 \
  -o results/ch04/hist_global ./build/release/bin/baseline_bench

python chapters/ch04_profiling/python/ncu_summary.py \
  --kernel "hist_" --count 4 ./build/release/bin/hist_bench
```

The optimization benchmark and report:

```bash
./build/release/bin/hist_bench > results/ch04/hist.csv
python chapters/ch04_profiling/python/perf_report.py results/ch04/hist.csv
```

The report shows each variant against the global-atomics baseline and the CPU. Privatization should help the most. Warp aggregation should help mainly on the skewed input and may cost a little on uniform input. The benchmark exits with an error if any kernel produces a wrong histogram.

## Tests

```bash
pytest chapters/ch04_profiling -v
ctest --preset release --output-on-failure
```

The histogram tests demand exact equality with the CPU reference at sizes that straddle the 16-byte vector width and the warp and block boundaries.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ncu` reports a permission error | Run as admin, or set `NVreg_RestrictProfilingToAdminUsers=0` |
| Docker profiling fails | Add `--cap-add=SYS_ADMIN` |
| `nvtx3/nvtx3.hpp` missing | Install the NVTX headers or use `nvToolsExt.h` |
| ncu durations longer than benchmark times | It locks clocks and flushes caches by default |
| Noisy benchmark spread | Check throttle reasons in the telemetry output |