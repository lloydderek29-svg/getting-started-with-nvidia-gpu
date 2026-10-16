# Chapter 10: Deploying GPU Models with Triton Inference Server

This directory builds a complete inference service around the Chapter 9 engine: a versioned model repository with preprocessing and postprocessing, clients, tuning experiments, monitoring, and a load-tested acceptance check. Shared code is in `common/python/gpubook/serving/`. Run every command from the repository root.

## What Is Here

| Path | Purpose |
|---|---|
| `templates/preprocess/` | Python backend model that decodes and crops JPEGs |
| `templates/postprocess/` | Python backend model that returns the top five labels |
| `templates/pets_pipeline/` | Ensemble tying the three models together |
| `docker/Dockerfile.triton` | Triton image with Pillow added |
| `docker/docker-compose.yml` | Server definition and ports |
| `python/build_repository.py` | Assembles and validates the model repository |
| `python/client_compare.py` | JSON, binary HTTP, and gRPC compared |
| `python/client_ensemble.py` | Classifies image files through the pipeline |
| `python/tune_batching.py` | Dynamic batching and instance experiments |
| `python/monitor_server.py` | Live terminal view of server metrics |
| `python/load_test.py` | Open-loop load test of the model or pipeline |
| `python/service_check.py` | Acceptance test against service objectives |
| `configs/slo.yaml` | Service-level objectives |
| `configs/prometheus.yml` | Prometheus scrape configuration |

The engine build script is `scripts/build_engine_in_container.sh` at the repository root.

## Prerequisites

You need Docker with the NVIDIA runtime, the Chapter 9 ONNX file and `classes.json`, and the validation images at `data/pets/val`.

```bash
pip install -r requirements/serving.txt
pip install -e .
```

Choose a current Triton release tag of the form year dot month, and use the TensorRT container with the same tag to build the engine, because an engine must match the server's TensorRT version.

## Build and Start

```bash
export TRITON_TAG=YY.MM

bash scripts/build_engine_in_container.sh $TRITON_TAG fp16

python chapters/ch10_triton/python/build_repository.py \
  --engine chapters/ch09_tensorrt/models/pets_fp16_triton.plan \
  --onnx chapters/ch09_tensorrt/models/pets_fp32.onnx \
  --classes chapters/ch09_tensorrt/models/classes.json

docker compose -f chapters/ch10_triton/docker/docker-compose.yml \
  up -d --build
```

The build script validates the repository and prints any problems. Watch the server log until every model shows READY.

```bash
docker compose -f chapters/ch10_triton/docker/docker-compose.yml \
  logs triton | grep -E "READY|UNAVAILABLE|error"

curl -s -o /dev/null -w "%{http_code}\n" localhost:8000/v2/health/ready
```

A response of 200 means the server and its models are ready. The ports are 8000 for HTTP, 8001 for gRPC, and 8002 for metrics. Never expose these ports, especially the model-control API, to untrusted networks.

## Use the Service

```bash
python chapters/ch10_triton/python/client_ensemble.py \
  data/pets/val/Abyssinian/*.jpg

python chapters/ch10_triton/python/client_compare.py
```

Expect top labels with probabilities for each image. In the protocol comparison, JSON is slowest, binary HTTP is much faster, and gRPC is close to binary HTTP for single requests and ahead under concurrency.

## Tune

```bash
python chapters/ch10_triton/python/tune_batching.py
```

The script tries five scheduling configurations with 32 concurrent clients, rewrites the model configuration, reloads it, and restores the original at the end. The table shows throughput, p50 and p99 latency, the average batch actually formed, and server-side queue and compute time. The repository directory must be writable for this script, so run it in development only.

## Monitor

```bash
python chapters/ch10_triton/python/monitor_server.py
```

Run this beside a load test. A batch column stuck near 1.0 under heavy load means batching is not engaging, and a rising pending count means arrivals exceed capacity. For continuous monitoring, point Prometheus at `configs/prometheus.yml`.

## Load Test and Acceptance

```bash
python chapters/ch10_triton/python/load_test.py \
  --target model --rates 100,200,400,800,1600

python chapters/ch10_triton/python/load_test.py \
  --target pipeline --rates 50,100,200,400,800

python chapters/ch10_triton/python/service_check.py
```

The load test sends requests on a Poisson schedule and measures latency from the intended send time, which avoids coordinated omission. It reports capacity as the highest rate meeting the latency and error targets, and the server breakdown shows which model is the bottleneck. The pipeline capacity is typically lower than the model capacity because JPEG decoding limits it, and adding preprocess instances helps until the CPU cores are used up.

Edit `configs/slo.yaml` to your real objectives before running the acceptance check. It tests accuracy against the Chapter 9 reference, latency and errors at the target rate, capacity, GPU memory, and failure handling for corrupt images, wrong shapes, and unknown models. The exit code is nonzero on failure, and the report is written to `results/ch10/service_check.json`. Use the corrected `service_accuracy` function from the chapter, which contains no stray assignment line.

To see overload behavior, rebuild with `--max-queue 64 --timeout-us 50000`, reload, and drive the service at two to three times capacity. Errors should rise while accepted requests keep their latency.

## Stop

```bash
docker compose -f chapters/ch10_triton/docker/docker-compose.yml down
```

## Tests

```bash
pytest chapters/ch10_triton -v
```

The tests cover the repository validator, configuration rendering, client retries, metrics parsing, capacity arithmetic, the load generator, and the model templates. None needs a running server or a GPU.

## Troubleshooting

| Problem | Fix |
|---|---|
| Engine incompatible at load | Rebuild in the TensorRT container with the same tag |
| Python models fail to load | Increase `shm_size` and confirm Pillow is installed |
| Model stays UNAVAILABLE | Read the reason in the startup table in the log |
| Client is the bottleneck | Use several client processes or `perf_analyzer` |
| Accuracy gap above one point | Preprocessing no longer matches training |
| Failures only in preprocess | Clients are sending bad images |