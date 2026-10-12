import argparse
import gc
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from torchvision.models import resnet50
	
from gpubook.bench import bench_cpu
from gpubook.trt.onnx_tools import Preprocessed, compare_outputs
from gpubook.trt.runtime import GraphedTrt, TrtModule

MODELS = Path("chapters/ch09_tensorrt/models")


def load_torch(checkpoint: str):
    blob = torch.load(checkpoint, map_location="cpu")
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(blob["classes"]))
    model.load_state_dict(blob["model"])
    return Preprocessed(model).eval().cuda()


class TorchBackend:
    def __init__(self, model, dtype, channels_last=False):
        self.dtype = dtype
        self.model = model.to(dtype)
        if channels_last:
            self.model = self.model.to(
                memory_format=torch.channels_last)

    def __call__(self, x):
        with torch.inference_mode():
            return self.model(x)


class OrtBackend:
    def __init__(self, path, dtype, num_classes):
        self.dtype = dtype
        self.np_dtype = (np.float16 if dtype == torch.float16
                         else np.float32)
        self.num_classes = num_classes
        self.sess = ort.InferenceSession(
            str(path),
            providers=[("CUDAExecutionProvider",
                        {"device_id": 0}),
                       "CPUExecutionProvider"])
        if self.sess.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError(
                "ONNX Runtime fell back to the CPU; check that "
                "onnxruntime-gpu matches your CUDA version")
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name

    def __call__(self, x):
        out = torch.empty((x.shape[0], self.num_classes),
                          dtype=x.dtype, device=x.device)
        b = self.sess.io_binding()
        b.bind_input(self.in_name, "cuda", 0, self.np_dtype,
                     tuple(x.shape), x.data_ptr())
        b.bind_output(self.out_name, "cuda", 0, self.np_dtype,
                      tuple(out.shape), out.data_ptr())
        self.sess.run_with_iobinding(b)
        b.synchronize_outputs()
        return out


class TrtBackend:
    def __init__(self, engine_path, graph=False):
        self.module = TrtModule(engine_path)
        self.dtype = self.module.input_dtype
        self.graph = graph
        self._graphs = {}

    def __call__(self, x):
        if not self.graph:
            return self.module(x)
        key = x.shape[0]
        if key not in self._graphs:
            self._graphs[key] = GraphedTrt(self.module, x)
        return self._graphs[key](x)


def used_gib() -> float:
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 2**30


def newest(pattern: str) -> Path:
    found = sorted(MODELS.glob(pattern))
    if not found:
        raise SystemExit(f"no file matching {pattern} in {MODELS}")
    return found[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="results/ch06/pets.pt")
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--classes", type=int, default=37)
    a = ap.parse_args()
    batches = [int(b) for b in a.batches.split(",")]

    reference = load_torch(a.checkpoint)
    ref_in = torch.rand(8, 3, 224, 224, device="cuda")
    with torch.inference_mode():
        ref_out = reference(ref_in).float().cpu().numpy()

    makers = {
        "torch fp32": lambda: TorchBackend(
            load_torch(a.checkpoint), torch.float32),
        "torch fp16 cl": lambda: TorchBackend(
            load_torch(a.checkpoint), torch.float16, True),
        "ort fp32": lambda: OrtBackend(
            MODELS / "pets_fp32.onnx", torch.float32, a.classes),
        "ort fp16": lambda: OrtBackend(
            MODELS / "pets_fp16.onnx", torch.float16, a.classes),
        "trt fp32": lambda: TrtBackend(newest("pets_fp32_*.engine")),
        "trt fp16": lambda: TrtBackend(newest("pets_fp16_*.engine")),
        "trt fp16 graph": lambda: TrtBackend(
            newest("pets_fp16_*.engine"), graph=True),
    }

    rows, base = [], {}
    for name, make in makers.items():
        gc.collect()
        torch.cuda.empty_cache()
        before = used_gib()
        try:
            backend = make()
            x8 = ref_in.to(backend.dtype)
            backend(x8)  # first call allocates
            torch.cuda.synchronize()
            mem = used_gib() - before
            agree = compare_outputs(
                ref_out, backend(x8).float().cpu().numpy())
        except Exception as err:
            print(f"{name:16s} unavailable: {err}")
            continue
        for b in batches:
            x = torch.rand(b, 3, 224, 224, device="cuda",
                           dtype=backend.dtype)
            s = bench_cpu(lambda: backend(x), warmup=20,
                          repeats=100, sync_cuda=True)
            base.setdefault(b, s.median_ms)
            rows.append((name, b, s, mem, agree))
        del backend

    print(f"{'backend':16s} {'batch':>5s} {'p50 ms':>8s} "
          f"{'p95 ms':>8s} {'img/s':>8s} {'speedup':>8s} "
          f"{'GiB':>6s} {'top1 agree':>11s}")
    for name, b, s, mem, agree in rows:
        print(f"{name:16s} {b:5d} {s.median_ms:8.3f} "
              f"{s.p95_ms:8.3f} {b / (s.median_ms / 1e3):8.0f} "
              f"{base[b] / s.median_ms:7.2f}x {mem:6.2f} "
              f"{agree['top1_agree']:11.3f}")


if __name__ == "__main__":
    main()
