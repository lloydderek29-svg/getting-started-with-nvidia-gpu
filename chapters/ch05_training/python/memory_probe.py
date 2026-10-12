from dataclasses import replace

import torch
import torch.nn as nn

from gpubook.training.config import TrainConfig
from gpubook.training.engine import train_one_epoch
from gpubook.training.model import resnet18_cifar


def probe(cfg: TrainConfig, steps: int = 20):
    dev = torch.device("cuda")
    torch.manual_seed(0)
    model = resnet18_cifar(
        grad_checkpoint=cfg.grad_checkpoint).to(dev)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=cfg.amp == "fp16")
    crit = nn.CrossEntropyLoss()

    micro = cfg.batch_size
    x = torch.randn(micro, 3, 32, 32, device=dev)
    y = torch.randint(0, 10, (micro,), device=dev)
    loader = [(x, y)] * steps

    train_one_epoch(model, loader, crit, opt, None,
                    dev, cfg, scaler)  # warm-up
    torch.cuda.reset_peak_memory_stats()
    r = train_one_epoch(model, loader, crit, opt, None,
                        dev, cfg, scaler)
    peak = torch.cuda.max_memory_allocated() / 2**30
    del model, opt
    torch.cuda.empty_cache()
    return r["images_per_s"], peak


def main() -> None:
    torch.set_float32_matmul_precision("high")
    base = TrainConfig(batch_size=256)
    cases = [
        ("fp32 (tf32 matmul)", {}),
        ("fp16 + scaler", dict(amp="fp16")),
        ("bf16", dict(amp="bf16")),
        ("bf16 + channels_last",
         dict(amp="bf16", channels_last=True)),
        ("bf16 + cl + checkpoint",
         dict(amp="bf16", channels_last=True,
              grad_checkpoint=True)),
        ("bf16 + cl, batch 1024",
         dict(amp="bf16", channels_last=True,
              batch_size=1024)),
        ("bf16 + cl + ckpt, batch 1024",
         dict(amp="bf16", channels_last=True,
              grad_checkpoint=True, batch_size=1024)),
    ]
    print(f"{'setting':32s} {'img/s':>9s} {'peak GiB':>9s}")
    for name, changes in cases:
        try:
            rate, peak = probe(replace(base, **changes))
            print(f"{name:32s} {rate:9.0f} {peak:9.2f}")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"{name:32s} {'out of memory':>19s}")


if __name__ == "__main__":
    main()
