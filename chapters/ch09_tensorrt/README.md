# Chapter 9: TensorRT, ONNX, and Optimized Model Inference

This directory exports the fine-tuned classifier from Chapter 6 to ONNX, builds TensorRT engines, compares backends, and applies a release gate. Shared code is in `common/python/gpubook/trt/`. Run every command from the repository root.

## What Is Here

| Path | Purpose |
|---|---|
| `python/export_classifier.py` | Exports to ONNX and validates the export |
| `python/build_engines.py` | Builds FP32, FP16, and BF16 engines |
| `python/smoke_engine.py` | Quick check that an engine runs |
| `python/inspect_engine.py` | Counts fused layers |
| `python/trt_batch_sweep.py` | Latency and throughput by batch, eager and graph |
| `python/compare_backends.py` | PyTorch, ONNX Runtime, and TensorRT benchmark |
| `python/deploy_check.py` | Accuracy, latency, throughput, memory, and stability gates |
| `configs/gates.yaml` | Gate thresholds |
| `configs/triton/pets/config.pbtxt` | Example Triton configuration |
| `models/` | Generated artifacts, not committed |

## Prerequisites

This chapter needs the checkpoint from Chapter 6 at `results/ch06/pets.pt` and the dataset at `data/pets`. Install the deployment packages and match the TensorRT wheel to your CUDA major version. The code targets the TensorRT 10 API.

```bash
pip install -r requirements/torch.txt
pip install -r requirements/deploy.txt
pip install -e .
```

## Export and Validate

```bash
python chapters/ch09_tensorrt/python/export_classifier.py

python chapters/ch09_tensorrt/python/export_classifier.py \
  --half --provider CUDAExecutionProvider
```

The first run writes `models/pets_fp32.onnx` and `classes.json`. The second writes `pets_fp16.onnx`. Each run compares ONNX Runtime with PyTorch at batch sizes 1, 4, and 13, and prints the operator histogram. For FP32 expect differences around 1e-5, a cosine near one, and perfect agreement. The FP16 export has larger differences. This check uses random inputs and proves structure only, and real accuracy is tested in the gate.

## Build Engines

```bash
python chapters/ch09_tensorrt/python/build_engines.py
```

Engines are named by model, precision, GPU architecture, TensorRT version, and batch profile, and a JSON metadata file sits beside each. A timing cache speeds up rebuilds. An engine is valid only for the GPU model and TensorRT version that built it. Use `--detailed` when you want the layer inspection below.

```bash
ENGINE=$(ls chapters/ch09_tensorrt/models/pets_fp16_sm*.engine | head -n1)

python chapters/ch09_tensorrt/python/smoke_engine.py $ENGINE

python chapters/ch09_tensorrt/python/inspect_engine.py \
  $ENGINE chapters/ch09_tensorrt/models/pets_fp32.onnx

python chapters/ch09_tensorrt/python/trt_batch_sweep.py $ENGINE
```

The inspection should show far fewer engine layers than ONNX nodes. The sweep shows cost per image falling with batch size, with graph replay helping most at batch one. If graph capture fails on your TensorRT version, the sweep falls back to eager and says so.

## Compare Backends

```bash
python chapters/ch09_tensorrt/python/compare_backends.py --batches 1,8,32
```

The seven rows are PyTorch FP32, PyTorch FP16 with channels-last, ONNX Runtime FP32 and FP16, and TensorRT FP32, FP16, and FP16 with a CUDA graph. Columns show p50 and p95 latency, throughput, speedup over PyTorch FP32, memory, and top-1 agreement with the reference. A backend that fails is reported and skipped.

## Release Gate

Edit thresholds in `configs/gates.yaml` from your own baseline, then run:

```bash
python chapters/ch09_tensorrt/python/deploy_check.py \
  --engine $ENGINE --graph
```

The gate measures accuracy drop and top-1 agreement on the real validation set, p99 latency at batch one, throughput at batch 32, memory, and drift and leak behavior over a soak run. It also checks finite outputs, determinism, rejection of an oversized batch, and a TensorRT version match. The exit code is nonzero on any failure, and the full report is written to `results/ch09/deploy_check.json`.

## Tests

```bash
pytest chapters/ch09_tensorrt -v
```

## Troubleshooting

| Problem | Fix |
|---|---|
| ONNX parse fails | Read the named operator, raise the opset, or rewrite it |
| Engine will not deserialize | GPU or TensorRT version differs from the build |
| ONNX Runtime uses the CPU | Install `onnxruntime-gpu` matching your CUDA version |
| Input shape rejected | Batch is outside the optimization profile |
| Large p99 above the median | Check throttling and other GPU processes |
| Models over 2 GB | Save the ONNX file with external data |