import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from torchvision.models import resnet50

from gpubook.trt.onnx_tools import (
    Preprocessed, compare_outputs, export_onnx, op_histogram)


def build(checkpoint: str):
    blob = torch.load(checkpoint, map_location="cpu")
    classes = blob["classes"]
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(blob["model"])
    return Preprocessed(model), classes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="results/ch06/pets.pt")
    ap.add_argument("--out", default="chapters/ch09_tensorrt/models")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--provider", default="CPUExecutionProvider")
    a = ap.parse_args()

    model, classes = build(a.checkpoint)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = "fp16" if a.half else "fp32"
    path = out / f"pets_{tag}.onnx"

    dev = "cuda" if a.half else "cpu"
    dtype = torch.float16 if a.half else torch.float32
    model = model.to(dev, dtype)
    example = torch.rand(2, 3, 224, 224, device=dev, dtype=dtype)
    export_onnx(model, example, path)
    onnx.checker.check_model(onnx.load(str(path)))
    (out / "classes.json").write_text(json.dumps(classes))
    print(f"exported {path} "
          f"({path.stat().st_size / 2**20:.1f} MiB)")

    sess = ort.InferenceSession(
        str(path), providers=[a.provider])
    print("providers:", sess.get_providers())
    name = sess.get_inputs()[0].name
    rng = np.random.default_rng(0)
    print(f"{'batch':>5s} {'max abs':>9s} {'cosine':>9s} "
          f"{'top1 agree':>11s}")
    for batch in (1, 4, 13):
        x = rng.random((batch, 3, 224, 224)).astype(
            np.float16 if a.half else np.float32)
        with torch.no_grad():
            ref = model(torch.from_numpy(x).to(dev)).float()
        got = sess.run(None, {name: x})[0]
        r = compare_outputs(ref.cpu().numpy(), got)
        print(f"{batch:5d} {r['max_abs']:9.2e} "
              f"{r['cosine_min']:9.6f} {r['top1_agree']:11.2f}")

    print("\noperator counts:")
    for op, n in op_histogram(path).most_common(8):
        print(f"  {op:22s} {n}")


if __name__ == "__main__":
    main()
