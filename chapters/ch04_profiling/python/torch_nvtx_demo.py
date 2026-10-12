import argparse
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch.cuda import nvtx, profiler


@contextmanager
def rng(name):
    nvtx.range_push(name)
    try:
        yield
    finally:
        nvtx.range_pop()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync-every-step", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cuda")
    model = nn.Sequential(
        nn.Linear(1024, 4096), nn.ReLU(),
        nn.Linear(4096, 4096), nn.ReLU(),
        nn.Linear(4096, 10),
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    x_cpu = torch.randn(512, 1024).pin_memory()
    y_cpu = torch.randint(0, 10, (512,)).pin_memory()

    def step():
        with rng("h2d"):
            x = x_cpu.to(dev, non_blocking=True)
            y = y_cpu.to(dev, non_blocking=True)
        with rng("forward"):
            loss = loss_fn(model(x), y)
        with rng("backward"):
            opt.zero_grad(set_to_none=True)
            loss.backward()
        with rng("optimizer"):
            opt.step()
        return loss

    for _ in range(10):
        step()
    torch.cuda.synchronize()

    profiler.start()
    for i in range(20):
        with rng(f"step_{i}"):
            loss = step()
            if args.sync_every_step:
                loss.item()  # forces a CPU/GPU sync
    torch.cuda.synchronize()
    profiler.stop()


if __name__ == "__main__":
    main()
