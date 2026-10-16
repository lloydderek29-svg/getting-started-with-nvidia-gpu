# Chapter 1: NVIDIA GPU Architecture and Performance Planning

This directory holds the code for inspecting a GPU, measuring its real memory bandwidth, PCIe throughput, and power behavior, planning VRAM for a model, and producing a capability report. Run every command from the repository root.

## What Is Here

| Path | Purpose |
|---|---|
| `src/device_properties.cu` | Prints every CUDA device property that matters for kernel design |
| `src/bandwidth_test.cu` | Measures device memory copy bandwidth against the computed peak |
| `src/transfer_test.cu` | Compares pageable and pinned host transfers over PCIe |
| `src/runtime_report.cu` | Emits one JSON line per GPU for the report tool |
| `python/torch_props.py` | Shows the properties PyTorch sees |
| `python/hardware_profile.py` | Detects ECC, MIG, and Jetson using NVML |
| `python/power_watch.py` | Samples power, clocks, and temperature once per second |
| `python/vram_planner.py` | Estimates inference and training VRAM for a model |
| `python/peak_memory.py` | Helper that measures peak training memory in PyTorch |
| `tests/test_report.py` | Tests for the report parser and warnings |

The shared code used here lives in `common/include/gpubook/` (error checking and timer) and `common/python/gpubook/report.py`.

## Prerequisites

You need an NVIDIA GPU of Turing generation or newer, a current driver, the CUDA Toolkit with `nvcc`, CMake 3.24 or newer, and Python 3.10 or newer. Install the Python packages once.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements/base.txt
pip install -r requirements/dev.txt
pip install -e .
```

The PyTorch scripts also need the deep learning stack.

```bash
pip install -r requirements/torch.txt
```

## Build

```bash
cmake --preset release
cmake --build --preset release -j
```

All executables land in `build/release/bin/`. The root `CMakeLists.txt` must list this chapter with `add_subdirectory(chapters/ch01_architecture)`.

## Run

Start with the device query, which needs nothing else.

```bash
./build/release/bin/device_properties
```

You should see one block per GPU. Illustrative output:

```text
Device 0: NVIDIA GeForce RTX 4070
  Compute capability   : 8.9
  Streaming MPs (SMs)  : 46
  Global memory        : 11.99 GiB
  Shared mem / block   : 48 KiB
  Shared mem / SM      : 100 KiB
```

Measure memory bandwidth and PCIe transfers.

```bash
./build/release/bin/bandwidth_test
./build/release/bin/transfer_test
```

Expect the bandwidth test to report a measured figure at roughly 85 to 95 percent of the computed peak. For GDDR6X and newer memory, treat the computed peak as a sanity bound and trust the measured number. In the transfer test, pinned transfers should clearly beat pageable ones.

Print the runtime view and the Python views.

```bash
./build/release/bin/runtime_report
python chapters/ch01_architecture/python/torch_props.py
python chapters/ch01_architecture/python/hardware_profile.py
```

Watch power and clocks for ten seconds. Start a workload in a second terminal first, such as the bandwidth test in a loop, or the clock will simply sit idle.

```bash
python chapters/ch01_architecture/python/power_watch.py
```

Plan VRAM for a model. The defaults describe an 8B-class model.

```bash
python chapters/ch01_architecture/python/vram_planner.py --vram 24
```

```text
Model: custom (8.0B params)
  Inference fp16 :   17.4 GiB  fits
  Inference 8-bit:   10.0 GiB  fits
  Inference 4-bit:    6.2 GiB  fits
  Full fine-tune :  119.2 GiB (before activations)
```

Build the capability report, which merges `nvidia-smi` and the runtime probe.

```bash
python -m gpubook.report --json gpu_report.json
```

If the probe is in a different location, pass it with `--runtime path/to/runtime_report`. Without the probe, the report still works from `nvidia-smi` alone and omits the runtime section.

## Tests

```bash
pytest chapters/ch01_architecture -v
```

The tests exercise parsing and warning logic, so they run on any machine, including one without a GPU.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `nvcc: command not found` | Add the toolkit `bin` directory to `PATH`, as in Chapter 2 |
| CMake cannot find CUDA | Check `nvcc --version` and `CUDA_HOME` |
| `None` for ECC or MIG | Normal on GeForce cards, the feature is unsupported |
| Low PCIe link width at idle | Links downshift when idle, query again under load |
| Bandwidth far below peak | Check throttling with `nvidia-smi -q -d PERFORMANCE` |