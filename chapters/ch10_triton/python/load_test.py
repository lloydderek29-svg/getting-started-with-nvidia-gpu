import argparse
import asyncio
import itertools
import json
import random
from pathlib import Path

import numpy as np
import tritonclient.grpc.aio as aio
import tritonclient.http as httpclient

from gpubook.serving.client import load_image_bytes
from gpubook.serving.loadgen import find_capacity, step_load
from gpubook.serving.metrics import latency_breakdown

MODELS = ["pets_pipeline", "preprocess", "pets", "postprocess"]


def snapshot(http, models):
    out = {}
    for m in models:
        try:
            out[m] = http.get_inference_statistics(m)[
                "model_stats"][0]
        except Exception:
            pass
    return out


def build_call(client, target, data_dir, n_images, seed):
    if target == "model":
        x = np.random.rand(1, 3, 224, 224).astype(np.float32)
        inp = aio.InferInput("input", list(x.shape), "FP32")
        inp.set_data_from_numpy(x)
        outs = [aio.InferRequestedOutput("logits")]

        async def call():
            await client.infer("pets", [inp], outputs=outs)
        return call

    files = sorted(Path(data_dir).rglob("*.jpg"))
    random.Random(seed).shuffle(files)
    inputs = []
    for f in files[:n_images]:
        arr = load_image_bytes([f])
        inp = aio.InferInput("image", list(arr.shape), "BYTES")
        inp.set_data_from_numpy(arr)
        inputs.append(inp)
    outs = [aio.InferRequestedOutput("top_probs"),
            aio.InferRequestedOutput("top_labels")]
    counter = itertools.count()

    async def call():
        inp = inputs[next(counter) % len(inputs)]
        await client.infer("pets_pipeline", [inp], outputs=outs)
    return call


async def run(a, http):
    client = aio.InferenceServerClient(url=a.grpc)
    call = build_call(client, a.target, a.data, a.images, a.seed)
    rows = []
    try:
        for rate in [float(r) for r in a.rates.split(",")]:
            before = snapshot(http, MODELS)
            part = await step_load(
                call, [rate], a.seconds, seed=a.seed)
            after = snapshot(http, MODELS)
            row = part[0]
            row["server"] = {
                m: latency_breakdown(before[m], after[m])
                for m in after if m in before}
            rows.append(row)
            print(f"{rate:7.0f} rps: p50 {row['p50_ms']:6.1f}"
                  f"  p99 {row['p99_ms']:7.1f}  err "
                  f"{100 * row['error_rate']:5.2f} %")
    finally:
        await client.close()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["model", "pipeline"],
                    default="pipeline")
    ap.add_argument("--rates", default="50,100,200,400,800")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--data", default="data/pets/val")
    ap.add_argument("--images", type=int, default=200)
    ap.add_argument("--http", default="localhost:8000")
    ap.add_argument("--grpc", default="localhost:8001")
    ap.add_argument("--p99-ms", type=float, default=40.0)
    ap.add_argument("--max-error", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    http = httpclient.InferenceServerClient(a.http)
    rows = asyncio.run(run(a, http))
    cap = find_capacity(rows, a.p99_ms, a.max_error)
    print(f"\ncapacity within p99 {a.p99_ms} ms and error "
          f"{a.max_error}: {cap:.0f} rps")
    last = rows[-1]["server"]
    print(f"{'model':16s} {'queue':>7s} {'infer':>7s} "
          f"{'batch':>6s}   (at the highest rate)")
    for m, bd in last.items():
        print(f"{m:16s} {bd['queue_ms']:7.2f} "
              f"{bd['compute_infer_ms']:7.2f} "
              f"{bd['avg_batch']:6.1f}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"rows": rows, "capacity_rps": cap}, indent=2))


if __name__ == "__main__":
    main()
