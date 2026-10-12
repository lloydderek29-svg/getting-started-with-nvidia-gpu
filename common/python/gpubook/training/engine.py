from __future__ import annotations

import time

import torch

_AMP_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


def _autocast(cfg):
    dtype = _AMP_DTYPES.get(cfg.amp)
    return torch.autocast(
        "cuda",
        dtype=dtype or torch.float16,
        enabled=dtype is not None,
    )


def _prepare(x, y, device, cfg):
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    if cfg.channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    return x, y


def train_one_epoch(model, loader, criterion, optimizer,
                    scheduler, device, cfg, scaler):
    model.train()
    accum = max(cfg.accum_steps, 1)
    n_batches = len(loader)
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    seen, wait, i = 0, 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    start = time.perf_counter()
    it = iter(loader)
    while True:
        t = time.perf_counter()
        try:
            x, y = next(it)
        except StopIteration:
            break
        wait += time.perf_counter() - t

        x, y = _prepare(x, y, device, cfg)
        if cfg.compile == "reduce-overhead":
            torch.compiler.cudagraph_mark_step_begin()

        with _autocast(cfg):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss / accum).backward()
        i += 1

        if i % accum == 0 or i == n_batches:
            if cfg.clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        loss_sum += loss.detach().float() * y.size(0)
        correct += (logits.argmax(1) == y).sum()
        seen += y.size(0)
        if cfg.max_steps and i >= cfg.max_steps:
            break

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "loss": float(loss_sum) / seen,
        "acc": float(correct) / seen,
        "seconds": elapsed,
        "images_per_s": seen / elapsed,
        "data_wait_s": wait,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, cfg):
    model.eval()
    loss_sum = torch.zeros((), device=device)
    top1 = torch.zeros((), device=device)
    top5 = torch.zeros((), device=device)
    n = 0
    for x, y in loader:
        x, y = _prepare(x, y, device, cfg)
        with _autocast(cfg):
            logits = model(x)
            loss = criterion(logits, y)
        loss_sum += loss.float() * y.size(0)
        top1 += (logits.argmax(1) == y).sum()
        k = min(5, logits.size(1))
        hits = logits.topk(k, dim=1).indices == y[:, None]
        top5 += hits.any(dim=1).sum()
        n += y.size(0)
    return {
        "loss": float(loss_sum) / n,
        "top1": float(top1) / n,
        "top5": float(top5) / n,
    }
