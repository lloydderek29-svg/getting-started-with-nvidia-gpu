from __future__ import annotations


def littles_law(throughput_rps: float, latency_s: float) -> float:
    """Average requests in flight."""
    return throughput_rps * latency_s


def capacity_rps(batch_ms: dict[int, float], batch: int,
                 instances: int = 1,
                 utilization: float = 0.7) -> float:
    """Rough requests per second at a steady batch size.

    An upper bound: instances share one GPU, so scaling
    beyond one instance is rarely linear."""
    exec_s = batch_ms[batch] / 1e3
    return instances * batch / exec_s * utilization


def largest_batch_within(batch_ms: dict[int, float],
                         budget_ms: float):
    ok = [b for b, ms in batch_ms.items() if ms <= budget_ms]
    return max(ok) if ok else None
# chapters/ch10_triton/tests/test_capacity.py
import pytest

from gpubook.serving.capacity import (
    capacity_rps, largest_batch_within, littles_law)

BATCH_MS = {1: 2.0, 8: 6.0, 32: 16.0, 64: 30.0}


def test_littles_law():
    assert littles_law(500, 0.02) == pytest.approx(10.0)


def test_capacity_grows_with_batch_size():
    small = capacity_rps(BATCH_MS, 1)
    large = capacity_rps(BATCH_MS, 32)
    assert small == pytest.approx(350.0)
    assert large == pytest.approx(1400.0)


def test_largest_batch_within_budget():
    assert largest_batch_within(BATCH_MS, 10.0) == 8
    assert largest_batch_within(BATCH_MS, 1.0) is None
