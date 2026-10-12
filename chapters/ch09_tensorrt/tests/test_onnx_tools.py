import numpy as np
import pytest
import torch
import torch.nn as nn

from gpubook.trt.onnx_tools import (
    Preprocessed, compare_outputs, export_onnx, op_histogram)


def test_identical_outputs_compare_perfectly():
    x = np.random.default_rng(0).normal(size=(6, 10))
    r = compare_outputs(x, x)
    assert r["max_abs"] == 0.0
    assert r["cosine_min"] == pytest.approx(1.0)
    assert r["top1_agree"] == 1.0


def test_flipped_prediction_is_detected():
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert compare_outputs(a, b)["top1_agree"] == 0.5


def test_preprocessing_matches_manual_normalization():
    model = nn.Identity()
    wrapped = Preprocessed(model, (0.5, 0.5, 0.5), (0.25,) * 3)
    x = torch.full((1, 3, 2, 2), 0.75)
    assert torch.allclose(wrapped(x), torch.ones(1, 3, 2, 2))


def test_exported_graph_contains_relu(tmp_path):
    net = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    path = tmp_path / "tiny.onnx"
    export_onnx(net, torch.randn(2, 4), path)
    assert op_histogram(path)["Relu"] == 1
