from __future__ import annotations

import json
import statistics
from pathlib import Path


def load_run(path) -> dict:
    return json.loads(Path(path).read_text())


def steady_throughput(history: list[dict]) -> float:
    rates = [e["train_images_per_s"] for e in history[1:]]
    if not rates:
        rates = [history[0]["train_images_per_s"]]
    return statistics.median(rates)


def time_to_target(history, target, key="val_top1"):
    elapsed = 0.0
    for e in history:
        elapsed += e["train_seconds"]
        if e[key] >= target:
            return elapsed
    return None


def summarize_run(run: dict, target: float = 0.9) -> dict:
    h = run["history"]
    images = sum(
        e["train_seconds"] * e["train_images_per_s"] for e in h)
    seconds = sum(e["train_seconds"] for e in h)
    waited = sum(e["train_data_wait_s"] for e in h)
    energy = (run.get("gpu") or {}).get("energy_j")
    return {
        "img_s": steady_throughput(h),
        "top1": h[-1]["val_top1"],
        "peak_gib": run["peak_mem_gib"],
        "data_wait_pct": 100.0 * waited / seconds,
        "j_per_kimg": (
            energy / (images / 1000.0) if energy else None),
        "t_target_s": time_to_target(h, target),
    }


def mean_std(values) -> tuple[float | None, float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None, 0.0
    mean = statistics.fmean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, std


def aggregate(runs: list[dict], target: float = 0.9) -> dict:
    rows = [summarize_run(r, target) for r in runs]
    return {k: mean_std([r[k] for r in rows]) for k in rows[0]}
