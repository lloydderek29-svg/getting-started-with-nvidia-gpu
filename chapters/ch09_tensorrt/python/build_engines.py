import argparse
from pathlib import Path

import torch

from gpubook.trt.build import (
    build_engine, engine_name, write_metadata)
import tensorrt as trt

MODELS = Path("chapters/ch09_tensorrt/models")
CACHE = MODELS / "timing.cache"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bmin", type=int, default=1)
    ap.add_argument("--bopt", type=int, default=8)
    ap.add_argument("--bmax", type=int, default=64)
    ap.add_argument("--detailed", action="store_true")
    a = ap.parse_args()

    gpu = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    cc = f"{major}.{minor}"

    jobs = [("fp32", "pets_fp32.onnx"),
            ("fp16", "pets_fp32.onnx"),
            ("bf16", "pets_fp32.onnx")]
    if not torch.cuda.is_bf16_supported():
        jobs.pop()

    for precision, onnx_file in jobs:
        name = engine_name("pets", precision, cc, trt.__version__,
                           a.bmin, a.bopt, a.bmax)
        path = MODELS / name
        info = build_engine(
            MODELS / onnx_file, path, precision=precision,
            min_shape=(a.bmin, 3, 224, 224),
            opt_shape=(a.bopt, 3, 224, 224),
            max_shape=(a.bmax, 3, 224, 224),
            timing_cache=CACHE, detailed=a.detailed)
        write_metadata(path, info, gpu, cc)
        print(f"{precision:5s} {info['size_mib']:7.1f} MiB  "
              f"built in {info['build_seconds']:6.1f} s  "
              f"{path.name}")


if __name__ == "__main__":
    main()
