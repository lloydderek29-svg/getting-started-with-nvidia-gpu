from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from .gpu_loader import GpuTensorLoader
from .repro import seed_worker

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


class SyntheticImages(Dataset):
    """Random images, a fixed pool reused cyclically."""

    def __init__(self, length=50000, shape=(3, 32, 32),
                 classes=10, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(256, *shape, generator=g)
        self.y = torch.randint(0, classes, (256,), generator=g)
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.x[i % 256], self.y[i % 256]


def _cifar_datasets(data_dir: str):
    norm = transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        norm,
    ])
    test_tf = transforms.Compose([transforms.ToTensor(), norm])
    train = datasets.CIFAR10(
        data_dir, train=True, download=True, transform=train_tf)
    test = datasets.CIFAR10(
        data_dir, train=False, download=True, transform=test_tf)
    return train, test


def build_loaders(cfg, device):
    """Return (train_loader, test_loader) for the config."""
    if cfg.loader == "gpu":
        return _build_gpu_loaders(cfg, device)

    if cfg.loader == "synthetic":
        train = SyntheticImages(50000)
        test = SyntheticImages(10000, seed=1)
    else:
        train, test = _cifar_datasets(cfg.data_dir)

    gen = torch.Generator().manual_seed(cfg.seed)
    kwargs = dict(
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        worker_init_fn=seed_worker,
    )
    if cfg.num_workers > 0:
        kwargs.update(
            persistent_workers=cfg.persistent_workers,
            prefetch_factor=cfg.prefetch_factor,
        )
    train_loader = DataLoader(
        train, batch_size=cfg.batch_size, shuffle=True,
        drop_last=True, generator=gen, **kwargs)
    test_loader = DataLoader(
        test, batch_size=512, shuffle=False, **kwargs)
    return train_loader, test_loader

def _cifar_to_gpu(ds, device):
    x = torch.from_numpy(ds.data).permute(0, 3, 1, 2)
    y = torch.tensor(ds.targets)
    return x.contiguous().to(device), y.to(device)


def _build_gpu_loaders(cfg, device):
    train = datasets.CIFAR10(cfg.data_dir, train=True,
                             download=True)
    test = datasets.CIFAR10(cfg.data_dir, train=False,
                            download=True)
    xtr, ytr = _cifar_to_gpu(train, device)
    xte, yte = _cifar_to_gpu(test, device)
    train_loader = GpuTensorLoader(
        xtr, ytr, cfg.batch_size, True,
        CIFAR_MEAN, CIFAR_STD, pad=4, seed=cfg.seed)
    test_loader = GpuTensorLoader(
        xte, yte, 512, False, CIFAR_MEAN, CIFAR_STD)
    return train_loader, test_loader
