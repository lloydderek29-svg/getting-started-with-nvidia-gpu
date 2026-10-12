from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import tensorrt as trt


def engine_name(model: str, precision: str, cc: str,
                trt_version: str, bmin: int, bopt: int,
                bmax: int) -> str:
    sm = cc.replace(".", "")
    return (f"{model}_{precision}_sm{sm}_trt{trt_version}"
            f"_b{bmin}-{bopt}-{bmax}.engine")


def build_engine(onnx_path, engine_path, *, precision="fp16",
                 input_name="input", min_shape, opt_shape,
                 max_shape, workspace_gib=4.0,
                 timing_cache=None, detailed=False,
                 verbose=False) -> dict:
    logger = trt.Logger(
        trt.Logger.INFO if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = [str(parser.get_error(i))
                  for i in range(parser.num_errors)]
        raise RuntimeError("ONNX parse failed:\n" + "\n".join(errors))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_gib * 2**30))
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "bf16":
        config.set_flag(trt.BuilderFlag.BF16)
    elif precision != "fp32":
        raise ValueError(f"unsupported precision {precision!r}")
    if detailed:
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    cache = None
    if timing_cache:
        path = Path(timing_cache)
        blob = path.read_bytes() if path.exists() else b""
        cache = config.create_timing_cache(blob)
        config.set_timing_cache(cache, False)

    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("engine build failed")
    seconds = time.perf_counter() - t0

    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(memoryview(serialized))
    if cache is not None:
        Path(timing_cache).write_bytes(memoryview(cache.serialize()))

    return {
        "engine": str(engine_path),
        "build_seconds": seconds,
        "size_mib": engine_path.stat().st_size / 2**20,
        "sha256": hashlib.sha256(
            engine_path.read_bytes()).hexdigest(),
        "trt_version": trt.__version__,
        "precision": precision,
        "profile": {"min": list(min_shape), "opt": list(opt_shape),
                    "max": list(max_shape)},
    }


def write_metadata(engine_path, info: dict, gpu: str,
                   cc: str) -> None:
    meta = dict(info, gpu=gpu, compute_capability=cc)
    Path(str(engine_path) + ".json").write_text(
        json.dumps(meta, indent=2))
