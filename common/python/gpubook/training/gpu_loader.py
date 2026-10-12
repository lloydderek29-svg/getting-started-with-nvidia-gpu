from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def random_crop_flip(x, out_h, out_w, pad=0):
    """Per-sample random crop and horizontal flip, batched."""
    n, c, _, _ = x.shape
    if pad:
        x = F.pad(x, (pad, pad, pad, pad))
    hp, wp = x.shape[-2:]
    dev = x.device
    i = torch.randint(0, hp - out_h + 1, (n,), device=dev)
    j = torch.randint(0, wp - out_w + 1, (n,), device=dev)
    rows = i[:, None] + torch.arange(out_h, device=dev)
    cols = j[:, None] + torch.arange(out_w, device=dev)
    flip = torch.rand(n, device=dev) < 0.5
    cols = torch.where(flip[:, None], cols.flip(1), cols)
    n_idx = torch.arange(n, device=dev)[:, None, None, None]
    c_idx = torch.arange(c, device=dev)[None, :, None, None]
    return x[n_idx, c_idx,
             rows[:, None, :, None], cols[:, None, None, :]]


def center_crop(x, out_h, out_w):
    h, w = x.shape[-2:]
    top, left = (h - out_h) // 2, (w - out_w) // 2
    return x[:, :, top:top + out_h, left:left + out_w]


class GpuTensorLoader:
    """Iterate over a dataset that lives entirely on one device."""

    def __init__(self, images, labels, batch_size, train,
                 mean, std, crop=None, pad=0, seed=0):
        self.images = images  # uint8, N x C x H x W
        self.labels = labels
        self.batch_size = batch_size
        self.train = train
        self.pad = pad
        h, w = images.shape[-2:]
        self.out_h, self.out_w = crop or (h, w)
        dev = images.device
        shape = (1, -1, 1, 1)
        self.mean = torch.tensor(
            mean, device=dev).view(shape) * 255
        self.std = torch.tensor(std, device=dev).view(shape) * 255
        self.gen = torch.Generator(device=dev)
        self.gen.manual_seed(seed)

    def __len__(self):
        n = len(self.images)
        if self.train:
            return n // self.batch_size
        return math.ceil(n / self.batch_size)

    def __iter__(self):
        n, dev = len(self.images), self.images.device
        if self.train:
            order = torch.randperm(n, device=dev,
                                   generator=self.gen)
        else:
            order = torch.arange(n, device=dev)
        bs = self.batch_size
        for k in range(len(self)):
            idx = order[k * bs:(k + 1) * bs]
            x = (self.images[idx].float() - self.mean) / self.std
            if self.train:
                x = random_crop_flip(
                    x, self.out_h, self.out_w, self.pad)
            elif (self.out_h, self.out_w) != x.shape[-2:]:
                x = center_crop(x, self.out_h, self.out_w)
            yield x.contiguous(), self.labels[idx]
