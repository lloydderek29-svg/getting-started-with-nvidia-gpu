"""Train ResNet-18 on CIFAR-10 with a configurable pipeline."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, fields
from pathlib import Path

import torch
import torch.nn as nn

from gpubook.env import snapshot
from gpubook.monitor import GpuMonitor
from gpubook.training.config import (
    TrainConfig, add_cli_args, load_config)
from gpubook.training.data import build_loaders
from gpubook.training.engine import evaluate, train_one_epoch
from gpubook.training.model import resnet18_cifar
from gpubook.training.repro import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    add_cli_args(parser)
    args = parser.parse_args()
    overrides = {
        f.name: getattr(args, f.name) for f in fields(TrainConfig)
    }
    cfg = load_config(args.config, overrides)

    seed_everything(cfg.seed, cfg.deterministic)
    device = torch.device("cuda")
    train_loader, test_loader = build_loaders(cfg, device)

    model = resnet18_cifar(
        grad_checkpoint=cfg.grad_checkpoint).to(device)
    if cfg.channels_last:  # Section 5.3
        model = model.to(memory_format=torch.channels_last)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
        nesterov=True, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(
        len(train_loader) / cfg.accum_steps)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, epochs=cfg.epochs,
        steps_per_epoch=steps_per_epoch)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(cfg.amp == "fp16"))  # Section 5.3

    eval_model = model
    if cfg.compile != "off":  # Section 5.4
        mode = None if cfg.compile == "default" else cfg.compile
        model = torch.compile(model, mode=mode)

    torch.cuda.reset_peak_memory_stats()
    history = []
    with GpuMonitor(interval=1.0) as mon:
        for epoch in range(cfg.epochs):
            tr = train_one_epoch(
                model, train_loader, criterion, optimizer,
                scheduler, device, cfg, scaler)
            ev = evaluate(
                model, test_loader, criterion, device, cfg)
            row = {"epoch": epoch}
            row.update({f"train_{k}": v for k, v in tr.items()})
            row.update({f"val_{k}": v for k, v in ev.items()})
            history.append(row)
            print(
                f"epoch {epoch:2d}  loss {tr['loss']:.3f}  "
                f"val top1 {ev['top1']:.4f}  "
                f"{tr['images_per_s']:8.0f} img/s  "
                f"data wait {tr['data_wait_s']:.1f}s")

    result = {
        "config": asdict(cfg),
        "env": snapshot(),
        "history": history,
        "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
        "gpu": mon.summary(),
    }
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{cfg.name}.json").write_text(
        json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
