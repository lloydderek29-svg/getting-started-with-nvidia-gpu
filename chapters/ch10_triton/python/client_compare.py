import argparse

import numpy as np
import tritonclient.grpc as grpcclient

from gpubook.bench import bench_cpu
from gpubook.serving.client import (
    infer_grpc, infer_http, make_http_client)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pets")
    ap.add_argument("--http", default="localhost:8000")
    ap.add_argument("--grpc", default="localhost:8001")
    a = ap.parse_args()

    http = make_http_client(a.http)
    grpc = grpcclient.InferenceServerClient(url=a.grpc)
    meta = http.get_model_metadata(a.model)
    print("model:", meta["name"],
          "inputs:", [i["name"] for i in meta["inputs"]])

    rng = np.random.default_rng(0)
    print(f"{'batch':>5s} {'json ms':>9s} "
          f"{'binary ms':>10s} {'grpc ms':>9s}")
    for b in (1, 8):
        x = rng.random((b, 3, 224, 224)).astype(np.float32)
        calls = {
            "json": lambda: infer_http(
                http, a.model, "input", x, ["logits"],
                binary=False),
            "binary": lambda: infer_http(
                http, a.model, "input", x, ["logits"],
                binary=True),
            "grpc": lambda: infer_grpc(
                grpc, a.model, "input", x, ["logits"]),
        }
        ref = calls["binary"]()[0]
        for name in ("json", "grpc"):
            got = calls[name]()[0]
            assert np.allclose(ref, got, atol=1e-4), name
        t = {k: bench_cpu(fn, warmup=5, repeats=30).median_ms
             for k, fn in calls.items()}
        print(f"{b:5d} {t['json']:9.1f} {t['binary']:10.1f} "
              f"{t['grpc']:9.1f}")


if __name__ == "__main__":
    main()
