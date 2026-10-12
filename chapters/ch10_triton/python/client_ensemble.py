import argparse
from pathlib import Path

import tritonclient.grpc as grpcclient

from gpubook.serving.client import infer_grpc, load_image_bytes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--url", default="localhost:8001")
    ap.add_argument("--top", type=int, default=3)
    a = ap.parse_args()

    client = grpcclient.InferenceServerClient(url=a.url)
    batch = load_image_bytes(a.images)
    probs, labels = infer_grpc(
        client, "pets_pipeline", "image", batch,
        ["top_probs", "top_labels"])
    for path, p_row, l_row in zip(a.images, probs, labels):
        pairs = ", ".join(
            f"{lab.decode()} {p:.2f}"
            for lab, p in zip(l_row[:a.top], p_row[:a.top]))
        print(f"{Path(path).name}: {pairs}")


if __name__ == "__main__":
    main()
