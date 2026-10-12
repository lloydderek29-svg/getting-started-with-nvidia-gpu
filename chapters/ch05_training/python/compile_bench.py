import sys
import time

import torch
import torch.nn as nn

from gpubook.bench import bench_cuda
from gpubook.training.model import resnet18_cifar


def build(mode: str, batch: int):
    dev = torch.device("cuda")
    model = resnet18_cifar().to(dev)
    model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    x = torch.randn(batch, 3, 32, 32, device=dev)
    x = x.contiguous(memory_format=torch.channels_last)
    y = torch.randint(0, 10, (batch,), device=dev)

    net = model
    if mode != "eager":
        net = torch.compile(
            model, mode=None if mode == "default" else mode)

    def step():
        if mode == "reduce-overhead":
            torch.compiler.cudagraph_mark_step_begin()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = crit(net(x), y)
        loss.backward()
        opt.step()

    return step


def main() -> None:
    batches = [int(b) for b in sys.argv[1:]] or [64, 512]
    modes = ["eager", "default", "reduce-overhead"]
    print(f"{'batch':>6s} {'mode':16s} {'first step':>11s} "
          f"{'steady':>9s} {'img/s':>9s}")
    for batch in batches:
        for mode in modes:
            torch.cuda.empty_cache()
            step = build(mode, batch)
            t0 = time.perf_counter()
            step()
            torch.cuda.synchronize()
            first = time.perf_counter() - t0
            s = bench_cuda(step, warmup=5, repeats=30)
            rate = batch / (s.median_ms / 1e3)
            print(f"{batch:6d} {mode:16s} {first:10.1f}s "
                  f"{s.median_ms:7.2f}ms {rate:9.0f}")


if __name__ == "__main__":
    main()
