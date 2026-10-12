import argparse
import asyncio
import json
import sys
import urllib.request
from pathlib import Path
	
import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
import yaml

from gpubook.serving.client import infer_grpc, load_image_bytes
from gpubook.serving.client import wait_ready
from gpubook.serving.metrics import gpu_summary, parse_prometheus
from gpubook.serving.slo import evaluate
from gpubook.vision.imagefolder import list_samples

sys.path.insert(0, str(Path(__file__).parent))
from load_test import build_call, snapshot  # noqa: E402
from gpubook.serving.loadgen import find_capacity, step_load  # noqa: E402
import tritonclient.grpc.aio as aio  # noqa: E402

MODELS = ["pets_pipeline", "preprocess", "pets", "postprocess"]


def service_accuracy(grpc, data_dir, batch=16):
    _, paths, labels = list_samples(Path(data_dir))
    correct = 0
    for s in range(0, len(paths), batch):
        chunk = paths[s:s + batch]
        _, top = infer_grpc(
            grpc, "pets_pipeline", "image",
            load_image_bytes(chunk),
            ["top_probs", "top_labels"])
        for p, lab in zip(chunk, top):
            correct += int(lab[0].decode() == p.parent.name)
    return correct / len(paths)


def failure_checks(grpc) -> dict:
    out = {}
    bad = np.empty((1, 1), dtype=np.object_)
    bad[0, 0] = b"not a jpeg"
    try:
        infer_grpc(grpc, "pets_pipeline", "image", bad,
                   ["top_probs"])
        out["rejects_corrupt_image"] = False
    except Exception:
        out["rejects_corrupt_image"] = True
    wrong = np.zeros((1, 3, 64, 64), dtype=np.float32)
    try:
        infer_grpc(grpc, "pets", "input", wrong, ["logits"])
        out["rejects_wrong_shape"] = False
    except Exception:
        out["rejects_wrong_shape"] = True
    try:
        infer_grpc(grpc, "no_such_model", "input", wrong,
                   ["logits"])
        out["rejects_unknown_model"] = False
    except Exception:
        out["rejects_unknown_model"] = True
    out["still_ready_after_errors"] = bool(
        grpc.is_server_ready()
        and grpc.is_model_ready("pets_pipeline"))
    return out


async def load_phase(a, http, rates):
    client = aio.InferenceServerClient(url=a.grpc)
    call = build_call(client, "pipeline", a.data, 200, 0)
    try:
        return await step_load(call, rates, a.seconds)
    finally:
        await client.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slo", default=(
        "chapters/ch10_triton/configs/slo.yaml"))
    ap.add_argument("--data", default="data/pets/val")
    ap.add_argument("--gate-report", default=(
        "results/ch09/deploy_check.json"))
    ap.add_argument("--http", default="localhost:8000")
    ap.add_argument("--grpc", default="localhost:8001")
    ap.add_argument("--metrics", default=(
        "http://localhost:8002/metrics"))
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--out", default=(
        "results/ch10/service_check.json"))
    a = ap.parse_args()

    http = httpclient.InferenceServerClient(a.http)
    grpc = grpcclient.InferenceServerClient(url=a.grpc)
    if not wait_ready(http, MODELS, timeout=60):
        print("server or models not ready")
        return 1

    slo = yaml.safe_load(Path(a.slo).read_text())
    target = float(slo.pop("target_rps"))
    m = {}

    acc = service_accuracy(grpc, a.data)
    ref_path = Path(a.gate_report)
    if ref_path.exists():
        ref = json.loads(ref_path.read_text())[
            "metrics"]["accuracy_ref"]
        m["accuracy_gap_pts"] = 100.0 * (ref - acc)
    print(f"service accuracy: {100 * acc:.2f} %")

    rates = sorted({target * f for f in
                    (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)})
    rows = asyncio.run(load_phase(a, http, rates))
    at_target = min(rows, key=lambda r: abs(r["rate"] - target))
    m["p99_ms_at_target"] = at_target["p99_ms"]
    m["error_rate_at_target"] = at_target["error_rate"]
    m["capacity_rps"] = find_capacity(
        rows, slo["max_p99_ms_at_target"],
        slo["max_error_rate_at_target"])

    with urllib.request.urlopen(a.metrics, timeout=3) as r:
        m["gpu_memory_gib"] = gpu_summary(
            parse_prometheus(r.read().decode()))["mem_gib"]

    checks = failure_checks(grpc)
    results = evaluate(m, slo)

    print(f"{'objective':28s} {'value':>10s} {'limit':>10s}")
    for r in results:
        v = "n/a" if r["value"] is None else f"{r['value']:.3f}"
        print(f"{r['gate']:28s} {v:>10s} {r['limit']:>10} "
              f"{'PASS' if r['ok'] else 'FAIL'}")
    for name, ok in checks.items():
        print(f"{name:28s} {'':>10s} {'':>10s} "
              f"{'PASS' if ok else 'FAIL'}")

    passed = (all(r["ok"] for r in results)
              and all(checks.values()))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"metrics": m, "checks": checks, "rows": rows,
         "passed": passed}, indent=2))
    print("\nresult:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
