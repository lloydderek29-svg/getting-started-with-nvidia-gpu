from __future__ import annotations

from pathlib import Path

import torch
import torchvision.transforms.v2.functional as TF
from torchvision.io import (
    ImageReadMode, decode_image, decode_jpeg, read_file)

from ..training.gpu_loader import GpuTensorLoader

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
JPEG = {".jpg", ".jpeg"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def list_samples(split_dir: Path):
    classes = sorted(p.name for p in split_dir.iterdir()
                     if p.is_dir())
    paths, labels = [], []
    for idx, name in enumerate(classes):
        for f in sorted((split_dir / name).rglob("*")):
            if f.suffix.lower() in EXTS:
                paths.append(f)
                labels.append(idx)
    return classes, paths, labels


def _resize_center(img: torch.Tensor, size: int):
    img = TF.resize(img, size, antialias=True)
    return TF.center_crop(img, [size, size])


def decode_files(paths, size: int, device, chunk: int = 64):
    out = []
    for s in range(0, len(paths), chunk):
        batch = paths[s:s + chunk]
        data = [read_file(str(p)) for p in batch]
        images = [None] * len(batch)

        jpg = [i for i, p in enumerate(batch)
               if p.suffix.lower() in JPEG]
        if device.type == "cuda" and jpg:
            try:
                dec = decode_jpeg(
                    [data[i] for i in jpg], device=device,
                    mode=ImageReadMode.RGB)
                for i, d in zip(jpg, dec):
                    images[i] = d
            except RuntimeError:
                pass  # fall back to the CPU below

        for i, d in enumerate(data):
            if images[i] is None:
                images[i] = decode_image(
                    d, mode=ImageReadMode.RGB).to(device)
        out.extend(_resize_center(im, size) for im in images)
    return torch.stack(out)


def load_split(root, split, size=256, device="cuda",
               cache=True):
    root, device = Path(root), torch.device(device)
    cache_file = root / f".cache_{split}_{size}.pt"
    if cache and cache_file.exists():
        blob = torch.load(cache_file)
        return (blob["x"].to(device), blob["y"].to(device),
                blob["classes"])

    classes, paths, labels = list_samples(root / split)
    x = decode_files(paths, size, device)
    y = torch.tensor(labels, device=device)
    if cache:
        torch.save({"x": x.cpu(), "y": y.cpu(),
                    "classes": classes}, cache_file)
    return x, y, classes


def build_image_loaders(root, size=256, crop=224,
                        batch_size=64, device="cuda", seed=0):
    xtr, ytr, classes = load_split(root, "train", size, device)
    xva, yva, val_classes = load_split(root, "val", size, device)
    if classes != val_classes:
        raise ValueError("train and val classes differ")
    train = GpuTensorLoader(
        xtr, ytr, batch_size, True, IMAGENET_MEAN,
        IMAGENET_STD, crop=(crop, crop), seed=seed)
    val = GpuTensorLoader(
        xva, yva, 128, False, IMAGENET_MEAN,
        IMAGENET_STD, crop=(crop, crop))
    return train, val, classes
