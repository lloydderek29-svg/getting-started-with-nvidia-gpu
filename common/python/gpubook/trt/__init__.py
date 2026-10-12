from __future__ import annotations

import inspect
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Preprocessed(nn.Module):
    """Accept images in 0..1 and normalize inside the model."""

    def __init__(self, model: nn.Module, mean=IMAGENET_MEAN,
                 std=IMAGENET_STD) -> None:
        super().__init__()
        self.model = model
        self.register_buffer(
            "mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer(
            "std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model((x - self.mean) / self.std)

def export_onnx(model: nn.Module, example: torch.Tensor, path,
                input_name: str = "input",
                output_name: str = "logits",
                opset: int = 17) -> None:
    kwargs = dict(
        input_names=[input_name],
        output_names=[output_name],
        dynamic_axes={input_name: {0: "batch"},
                      output_name: {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        torch.onnx.export(model, (example,), str(path), **kwargs)


def compare_outputs(ref: np.ndarray, test: np.ndarray) -> dict:
    ref = ref.astype(np.float64)
    test = test.astype(np.float64)
    diff = np.abs(ref - test)
    num = (ref * test).sum(axis=-1)
    den = (np.linalg.norm(ref, axis=-1)
           * np.linalg.norm(test, axis=-1) + 1e-12)
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "cosine_min": float((num / den).min()),
        "top1_agree": float(
            (ref.argmax(-1) == test.argmax(-1)).mean()),
    }


def op_histogram(onnx_path) -> Counter:
    import onnx

    model = onnx.load(str(onnx_path))
    return Counter(node.op_type for node in model.graph.node)
