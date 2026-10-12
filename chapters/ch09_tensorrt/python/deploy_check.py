from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import yaml

from gpubook.trt.build import trt
from gpubook.trt.gates import evaluate_gates
from gpubook.trt.runtime import GraphedTrt, TrtModule
from gpubook.training.gpu_loader import GpuTensorLoader
from gpubook.vision.imagefolder import load_split
from gpubook.vision.latency import percentile

sys.path.insert(0, str(Path(__file__).parent))
from compare_backends import load_torch  # noqa: E402


def validation_loader(root: str, batch: int):
    x, y, classes = load_split(root, "val", 256, "cuda")
    return GpuTensorLoader(
        x, y, batch, False, (0.0,) * 3, (1.0,) * 3,
        crop=(224, 224)), classes


@torch.no_grad()
def accuracy_and_agreement(ref, engine, loader):
    correct_ref = correct_eng = agree = n = 0
    for x, y in loader:
        r = ref(x).argmax(1)
        e = engine(x.to(engine.input_dtype)).argmax(1)
        correct_ref += int((r == y).sum())
        correct_eng += int((e == y).sum())
        agree += int((r == e).sum())
        n += y.numel()
    return correct_ref / n, correct_eng / n, agree / n


def time_calls(fn, iters: int, warmup: int = 30) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return times


def free_mib() -> float:
    return torch.cuda.mem_get_info()[0] / 2**20


@torch.no_grad()
def robustness(engine, max_batch: int) -> dict:
    dt = engine.input_dtype
    out = {"finite": True, "deterministic": True,
           "rejects_oversize": False, "batch_diff": 0.0}
    for value in (0.0, 1.0):
        x = torch.full((4, 3, 224, 224), value,
                       device="cuda", dtype=dt)
        out["finite"] &= bool(torch.isfinite(engine(x)).all())

    x = torch.rand(7, 3, 224, 224, device="cuda", dtype=dt)
    a = engine(x).clone()
    out["deterministic"] = bool(torch.equal(a, engine(x)))

    alone = engine(x[:1]).float()
    in_batch = a[:1].float()
    out["batch_diff"] = float((alone - in_batch).abs().max())

    for b in (1, 2, 3, 5, 17, max_batch):
        xb = torch.rand(b, 3, 224, 224, device="cuda", dtype=dt)
        out["finite"] &= bool(torch.isfinite(engine(xb)).all())

    try:
        engine(torch.rand(max_batch + 1, 3, 224, 224,
                          device="cuda", dtype=dt))
    except ValueError:
        out["rejects_oversize"] = True
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--checkpoint", default="results/ch06/pets.pt")
    ap.add_argument("--data", default="data/pets")
    ap.add_argument("--gates",
                    default="chapters/ch09_tensorrt/configs/gates.yaml")
    ap.add_argument("--soak", type=int, default=3000)
    ap.add_argument("--max-batch", type=int, default=64)
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--out", default="results/ch09/deploy_check.json")
    a = ap.parse_args()

    meta_path = Path(a.engine + ".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    ref = load_torch(a.checkpoint)

    before = free_mib()
    engine = TrtModule(a.engine)
    loader, _ = validation_loader(a.data, 64)
    m: dict = {}

    acc_ref, acc_eng, agree = accuracy_and_agreement(
        ref, engine, loader)
    m["accuracy_ref"] = acc_ref
    m["accuracy_engine"] = acc_eng
    m["accuracy_drop_pts"] = 100.0 * (acc_ref - acc_eng)
    m["top1_agreement"] = agree

    def runner(batch: int):
        x = torch.rand(batch, 3, 224, 224, device="cuda",
                       dtype=engine.input_dtype)
        if a.graph:
            g = GraphedTrt(engine, x)
            return lambda: g(x)
        return lambda: engine(x)

    lat1 = time_calls(runner(1), 500)
    m["p50_ms_b1"] = percentile(lat1, 0.50)
    m["p99_ms_b1"] = percentile(lat1, 0.99)

    lat32 = time_calls(runner(32), 200)
    m["throughput_b32"] = 32 / (statistics.median(lat32) / 1e3)

    big = runner(a.max_batch)
    big()
    torch.cuda.synchronize()
    m["memory_gib"] = (before - free_mib()) / 1024.0

    soak_fn = runner(8)
    soak = time_calls(soak_fn, a.soak)
    k = max(len(soak) // 5, 1)
    m["latency_drift"] = (statistics.median(soak[-k:])
                          / statistics.median(soak[:k]))
    mem_start = free_mib()
    time_calls(soak_fn, a.soak // 3, warmup=0)
    m["memory_growth_mib"] = max(mem_start - free_mib(), 0.0)

    rob = robustness(engine, a.max_batch)
    m["batch_invariance_diff"] = rob["batch_diff"]

    checks = {
        "outputs_finite": rob["finite"],
        "deterministic": rob["deterministic"],
        "rejects_out_of_profile_batch": rob["rejects_oversize"],
        "engine_matches_runtime": (
            not meta or meta.get("trt_version") == trt.__version__),
    }

    gates = yaml.safe_load(Path(a.gates).read_text())
    results = evaluate_gates(m, gates)

    print(f"engine: {Path(a.engine).name}")
    print(f"{'gate':28s} {'value':>10s} {'limit':>10s}  status")
    for r in results:
        v = "n/a" if r["value"] is None else f"{r['value']:.3f}"
        print(f"{r['gate']:28s} {v:>10s} {r['limit']:>10}  "
              f"{'PASS' if r['ok'] else 'FAIL'}")
    for name, ok in checks.items():
        print(f"{name:28s} {'':>10s} {'':>10s}  "
              f"{'PASS' if ok else 'FAIL'}")

    passed = (all(r["ok"] for r in results)
              and all(checks.values()))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"engine": a.engine, "metrics": m, "checks": checks,
         "gates": results, "passed": passed}, indent=2))
    print(f"\nresult: {'PASS' if passed else 'FAIL'}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
