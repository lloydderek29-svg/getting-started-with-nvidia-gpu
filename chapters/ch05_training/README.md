# Chapter 5: High-Performance Deep Learning with PyTorch

This directory trains ResNet-18 on CIFAR-10 with a configurable pipeline and measures the effect of each optimization on throughput, memory, energy, and accuracy. The training package lives in `common/python/gpubook/training/`. Run every command from the repository root.

## What Is Here

| Path | Purpose |
|---|---|
| `python/train.py` | Main training script, driven by a config and flags |
| `python/bench_loader.py` | Data loader throughput without a model |
| `python/bench_transfer.py` | Pageable, pinned, and uint8 transfer costs |
| `python/memory_probe.py` | Throughput and peak memory by precision and layout |
| `python/compile_bench.py` | Eager against `torch.compile` modes |
| `python/cuda_graph_step.py` | A full training step captured as a CUDA graph |
| `python/run_matrix.py` | Multi-seed experiment matrix and summary table |
| `configs/baseline.yaml` | Baseline experiment settings |
| `tests/` | Config, reproducibility, loader, accumulation, and report tests |

## Prerequisites

```bash
pip install -r requirements/torch.txt
pip install -e .
```

The first run downloads CIFAR-10 into `data/`.

## Run

The baseline:

```bash
python chapters/ch05_training/python/train.py \
  --config chapters/ch05_training/configs/baseline.yaml
```

Each epoch prints loss, validation accuracy, images per second, and the time spent waiting for data. After about 15 epochs expect validation accuracy in the high eighties to low nineties. A result file is written to `results/ch05/baseline.json`, including configuration, environment, per-epoch history, peak memory, and GPU telemetry.

Command-line flags override the file. Boolean flags take a value.

```bash
python chapters/ch05_training/python/train.py \
  --name gpu_bf16 --loader gpu --amp bf16 \
  --channels-last true --epochs 10
```

The main options:

| Flag | Values | Effect |
|---|---|---|
| `--loader` | `dataloader`, `gpu`, `synthetic` | Data path |
| `--num-workers` | integer | Loader processes |
| `--pin-memory` | true or false | Pinned batches |
| `--gpu-prefetch` | true or false | Side-stream prefetching |
| `--amp` | `off`, `fp16`, `bf16` | Mixed precision |
| `--accum-steps` | integer | Gradient accumulation |
| `--channels-last` | true or false | Memory format |
| `--grad-checkpoint` | true or false | Activation checkpointing |
| `--compile` | `off`, `default`, `reduce-overhead` | Compilation |
| `--deterministic` | true or false | Deterministic kernels |

Use FP16 with its scaler on Turing cards, and BF16 on Ampere and newer.

## Diagnostic Scripts

```bash
python chapters/ch05_training/python/bench_loader.py
python chapters/ch05_training/python/bench_transfer.py
python chapters/ch05_training/python/memory_probe.py
python chapters/ch05_training/python/compile_bench.py 64 512
python chapters/ch05_training/python/cuda_graph_step.py
```

Expect `bench_loader.py` to rank the GPU-resident loader first. Expect `memory_probe.py` to show mixed precision raising speed while lowering memory, and checkpointing lowering memory further at a speed cost. Expect `compile_bench.py` to show the largest gains at the small batch. Out-of-memory cases print as data points and do not abort the run.

## The Experiment Matrix

```bash
python chapters/ch05_training/python/run_matrix.py
```

This trains six configurations across three seeds, each in its own process, so it takes a while. It resumes after interruption by skipping result files that exist, and it prints a table with throughput, speedup, peak memory, accuracy with spread, energy per thousand images, and data wait. Compare accuracy differences against the printed standard deviation before calling one configuration better.

## Verify Reproducibility

```bash
for run in a b; do
  python chapters/ch05_training/python/train.py \
    --name det_$run --deterministic true \
    --epochs 1 --num-workers 0
done
```

The first-epoch training loss in the two JSON files should be identical.

## Tests

```bash
pytest chapters/ch05_training -v
```

All tests run on the CPU.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Unknown config keys` | A key in the YAML file or a flag has a typo |
| DataLoader worker crash in Docker | Increase `--shm-size` |
| Scheduler warning on early FP16 steps | Expected while the loss scale settles |
| `torch.compile` very slow at first | Compilation is cached for later runs |
| Out of memory | Lower `--batch-size`, raise `--accum-steps`, add `--amp bf16` |