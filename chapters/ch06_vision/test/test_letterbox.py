import torch

from gpubook.vision.letterbox import (
    boxes_to_frame, letterbox_gpu)


def test_shape_padding_and_content():
    frame = torch.full((100, 200, 3), 255, dtype=torch.uint8)
    out, r, (left, top) = letterbox_gpu(frame, size=64)
    assert out.shape == (1, 3, 64, 64)
    assert abs(r - 0.32) < 1e-6
    assert (left, top) == (0, 16)
    assert torch.allclose(out[0, 0, 32, 32], torch.tensor(1.0))
    pad = torch.tensor(114.0 / 255.0)
    assert torch.allclose(out[0, 0, 0, 0], pad)


def test_box_round_trip():
    r, pad = 0.32, (0, 16)
    box = torch.tensor([[40.0, 20.0, 120.0, 60.0]])
    lb = box.clone()
    lb[:, [0, 2]] = lb[:, [0, 2]] * r + pad[0]
    lb[:, [1, 3]] = lb[:, [1, 3]] * r + pad[1]
    back = boxes_to_frame(lb, r, pad, 200, 100)
    assert torch.allclose(back, box, atol=1e-4)
