import argparse
import asyncio
from pathlib import Path

import numpy as np
import tritonclient.grpc.aio as aio
import tritonclient.http as httpclient

from gpubook.serving.client import wait_ready
from gpubook.serving.loadgen import closed_loop, summarize
from gpubook.serving.metrics import latency_breakdown
from gpubook.serving.repo import BatchSettings, render_pets_config

CASES = [
    ("no batching", dict(dynamic=False)),
    ("delay 0", dict(preferred=(), max_delay_us=0)),
    ("delay 0.5 ms", dict(preferred=(), max_delay_us=500)),
    ("delay 2 ms, pref", dict(max_delay_us=2000)),
    ("2 inst, delay 2 ms", dict(max_delay_us=2000,
                                instances=2)),
]


async def drive(url, concurrency, seconds):
    client = aio.InferenceServerClient(url=url)
    x = np.random.rand(1, 3, 224, 224).astype(np.float32)
    inp = aio.InferInput("input", list(x.shape), "FP32")
    inp.set_data_from_numpy(x)
    out = aio.InferRequestedOutput("logits")

    async def call():
        await client.infer("pets", [inp], outputs=[out])

    try:
        return await closed_loop(call, concurrency, seconds)
    finally:
        await client.close()


def stats(http):
    return http.get_inference_statistics("pets")[
        "model_stats"][0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", default="localhost:8000")
    ap.add_argument("--grpc", default="localhost:8001")
    ap.add_argument("--repo", default=(
        "chapters/ch10_triton/model_repository"))
    ap.add_argument("--classes", type=int, default=37)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--seconds", type=float, default=15.0)
    a = ap.parse_args()

    http = httpclient.InferenceServerClient(a.http)
    cfg = Path(a.repo) / "pets" / "config.pbtxt"
    original = cfg.read_text()
    print(f"{'case':20s} {'req/s':>7s} {'p50':>7s} "
          f"{'p99':>7s} {'batch':>6s} {'queue':>7s} "
          f"{'infer':>7s}")
    try:
        for label, opts in CASES:
            cfg.write_text(render_pets_config(
                "pets", "tensorrt_plan",
                BatchSettings(**opts), a.classes))
            http.load_model("pets")
            wait_ready(http, ["pets"], timeout=120)
            before = stats(http)
            res = asyncio.run(
                drive(a.grpc, a.concurrency, a.seconds))
            bd = latency_breakdown(before, stats(http))
            s = summarize(res)
            print(f"{label:20s} {s['throughput']:7.0f} "
                  f"{s['p50_ms']:7.1f} {s['p99_ms']:7.1f} "
                  f"{bd['avg_batch']:6.1f} "
                  f"{bd['queue_ms']:7.2f} "
                  f"{bd['compute_infer_ms']:7.2f}")
    finally:
        cfg.write_text(original)
        http.load_model("pets")


if __name__ == "__main__":
    main()
