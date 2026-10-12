import importlib.util
import io
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).parents[1] / "templates"


def load(path):
    stub = types.ModuleType("triton_python_backend_utils")
    sys.modules["triton_python_backend_utils"] = stub
    spec = importlib.util.spec_from_file_location(
        path.parent.name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def jpeg(size, color):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG")
    return buf.getvalue()


def test_preprocess_shape_and_range_for_both_aspects():
    pre = load(ROOT / "preprocess" / "model.py")
    wide = pre.to_pixels(jpeg((300, 200), (255, 0, 0)))
    tall = pre.to_pixels(jpeg((200, 300), (255, 0, 0)))
    for arr in (wide, tall):
        assert arr.shape == (3, 224, 224)
        assert arr.dtype == np.float32
        assert 0.0 <= arr.min() and arr.max() <= 1.0
        assert arr[0].mean() > 0.9 and arr[1].mean() < 0.1


def test_preprocess_rejects_garbage():
    pre = load(ROOT / "preprocess" / "model.py")
    with pytest.raises(Exception):
        pre.to_pixels(b"not an image")


def test_softmax_topk_orders_and_normalizes():
    post = load(ROOT / "postprocess" / "model.py")
    logits = np.array([[0.0, 1.0, 2.0], [3.0, 0.0, 0.0]])
    probs, idx = post.softmax_topk(logits, k=2)
    assert idx[0].tolist() == [2, 1]
    assert idx[1][0] == 0
    assert np.all(np.diff(probs, axis=1) <= 0)
    assert probs.sum() < 2.0 and probs[0].sum() < 1.0
