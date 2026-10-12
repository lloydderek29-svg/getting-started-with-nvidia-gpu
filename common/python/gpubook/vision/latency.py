from __future__ import annotations

from collections import defaultdict


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("no values")
    v = sorted(values)
    pos = q * (len(v) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


class LatencyTracker:
    def __init__(self) -> None:
        self._data: dict[str, list[float]] = defaultdict(list)

    def add(self, stage: str, ms: float) -> None:
        self._data[stage].append(ms)

    def summary(self, skip: int = 0) -> dict:
        out = {}
        for stage, vals in self._data.items():
            vals = vals[skip:]
            if not vals:
                continue
            out[stage] = {
                "n": len(vals),
                "mean": sum(vals) / len(vals),
                "p50": percentile(vals, 0.50),
                "p95": percentile(vals, 0.95),
                "p99": percentile(vals, 0.99),
                "max": max(vals),
            }
        return out
