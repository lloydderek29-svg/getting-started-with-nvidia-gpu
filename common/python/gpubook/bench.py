"""Repeatable CPU and GPU benchmarking helpers."""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class Stats:
    n: int
    median_ms: float
    mean_ms: float
    std_ms: float
    p05_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float


def _pick(sorted_vals: list[float], q: float) -> float:
    idx = int(q * (len(sorted_vals) - 1) + 0.5)
    return sorted_vals[idx]


def summarize(samples: list[float]) -> Stats:
    if not samples:
        raise ValueError("no samples to summarize")
    v = sorted(samples)
    n = len(v)
    mean = sum(v) / n
    var = sum((x - mean) ** 2 for x in v) / max(n - 1, 1)
    mid = n // 2
    median = v[mid] if n % 2 else 0.5 * (v[mid - 1] + v[mid])
    return Stats(
        n=n,
        median_ms=median,
        mean_ms=mean,
        std_ms=math.sqrt(var),
        p05_ms=_pick(v, 0.05),
        p95_ms=_pick(v, 0.95),
        min_ms=v[0],
        max_ms=v[-1],
    )


def _cuda_sync() -> None:
    import torch

    torch.cuda.synchronize()


def bench_cpu(
    fn: Callable[[], object],
    warmup: int = 3,
    repeats: int = 20,
    sync_cuda: bool = False,
) -> Stats:
    """Wall-clock timing. Set sync_cuda for mixed CPU/GPU code."""
    sync = _cuda_sync if sync_cuda else (lambda: None)
    for _ in range(warmup):
        fn()
    sync()
    samples = []
    for _ in range(repeats):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        samples.append((time.perf_counter() - t0) * 1e3)
    return summarize(samples)


def bench_cuda(
    fn: Callable[[], object],
    warmup: int = 5,
    repeats: int = 30,
    flush_l2_mb: int = 0,
) -> Stats:
    """GPU timing with CUDA events on the current stream."""
    import torch

    flush = None
    if flush_l2_mb:
        flush = torch.empty(
            flush_l2_mb * 2**20, dtype=torch.uint8, device="cuda"
        )
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        if flush is not None:
            flush.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return summarize(samples)


def save_results(
    path: str | Path,
    results: dict[str, Stats],
    extra: dict | None = None,
) -> None:
    """Write results plus an environment snapshot to JSON."""
    from .env import snapshot

    payload = {
        "env": snapshot(),
        "results": {k: asdict(v) for k, v in results.items()},
        "extra": extra or {},
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2))
