#!/usr/bin/env bash

set -euo pipefail
TAG=${1:?usage: build_engine_in_container.sh TAG [fp16|fp32]}
PREC=${2:-fp16}
M=chapters/ch09_tensorrt/models
FLAG=""
[ "$PREC" = "fp16" ] && FLAG="--fp16"

docker run --gpus all --rm \
  --user "$(id -u):$(id -g)" \
  -v "$PWD":/work \
  "nvcr.io/nvidia/tensorrt:${TAG}-py3" \
  trtexec --onnx=/work/$M/pets_fp32.onnx \
    --saveEngine=/work/$M/pets_${PREC}_triton.plan \
    $FLAG \
    --minShapes=input:1x3x224x224 \
    --optShapes=input:8x3x224x224 \
    --maxShapes=input:64x3x224x224 \
    --memPoolSize=workspace:4096
