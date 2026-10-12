import time
from dataclasses import replace

import torch

from gpubook.training.config import TrainConfig
from gpubook.training.data import build_loaders

CASES = [
    ("workers=0", dict(num_workers=0)),
    ("workers=2", dict(num_workers=2)),
    ("workers=4", dict(num_workers=4)),
    ("workers=8 pinned", dict(
        num_workers=8, pin_memory=True,
        persistent_workers=True)),
    ("gpu resident", dict(loader="gpu")),
]


def measure(cfg, device, batches=100, warm=10) -> float:
    loader, _ = build_loaders(cfg, device)
    it = iter(loader)
    for _ in range(warm):
        x, _y = next(it)
        x.to(device, non_blocking=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    seen = 0
    for _ in range(batches):
        x, y = next(it)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        seen += y.size(0)
    torch.cuda.synchronize()
    return seen / (time.perf_counter() - t0)


def main() -> None:
    device = torch.device("cuda")
    base = TrainConfig()
    for name, changes in CASES:
        cfg = replace(base, **changes)
        rate = measure(cfg, device)
        print(f"{name:20s} {rate:10.0f} images/s")


if __name__ == "__main__":
    main()
