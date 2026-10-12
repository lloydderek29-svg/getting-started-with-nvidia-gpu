from __future__ import annotations

import torch
import torch.nn.functional as F


def letterbox_gpu(frame: torch.Tensor, size: int = 640,
                  pad_value: float = 114.0 / 255.0):
    """Resize a uint8 HxWx3 RGB frame to a size x size input.

    Returns the float tensor (1x3xSxS in 0..1), the scale
    factor, and the (left, top) padding in pixels.
    """
    h, w = frame.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    x = frame.permute(2, 0, 1)[None].float().div_(255.0)
    if (nh, nw) != (h, w):
        x = F.interpolate(
            x, size=(nh, nw), mode="bilinear",
            align_corners=False, antialias=r < 1.0)
    top, left = (size - nh) // 2, (size - nw) // 2
    out = x.new_full((1, 3, size, size), pad_value)
    out[:, :, top:top + nh, left:left + nw] = x
    return out, r, (left, top)


def boxes_to_frame(xyxy: torch.Tensor, r: float, pad,
                   width: int, height: int) -> torch.Tensor:
    """Map boxes from letterboxed coordinates to the frame."""
    left, top = pad
    out = xyxy.clone()
    out[:, [0, 2]] = (out[:, [0, 2]] - left) / r
    out[:, [1, 3]] = (out[:, [1, 3]] - top) / r
    out[:, [0, 2]] = out[:, [0, 2]].clamp(0, width)
    out[:, [1, 3]] = out[:, [1, 3]].clamp(0, height)
    return out
