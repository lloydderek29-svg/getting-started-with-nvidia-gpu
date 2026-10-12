import asyncio

import pytest

from gpubook.serving.loadgen import (
    LoadResult, closed_loop, find_capacity, open_loop,
    poisson_schedule, summarize)
from gpubook.serving.slo import evaluate


def test_poisson_schedule_rate_and_order():
    s = poisson_schedule(200, 20, seed=1)
    assert s == sorted(s) and s[-1] < 20
    assert abs(len(s) - 4000) < 240


def test_closed_loop_measures_latency():
    async def call():
        await asyncio.sleep(0.01)

    res = asyncio.run(closed_loop(call, 4, 0.3, warmup_s=0.1))
    assert len(res.latencies_ms) > 20
    assert min(res.latencies_ms) >= 9.0


def test_open_loop_accounts_for_every_request():
    state = {"n": 0}

    async def call():
        state["n"] += 1
        if state["n"] % 2 == 0:
            raise RuntimeError("boom")

    res = asyncio.run(open_loop(call, 200, 0.5))
    assert res.offered > 0
    done = res.errors + len(res.latencies_ms) + res.dropped
    assert done == res.offered


def test_summarize_percentiles_and_error_rate():
    res = LoadResult(latencies_ms=[1.0, 2.0, 3.0, 4.0],
                     errors=1, offered=5, seconds=2.0)
    s = summarize(res)
    assert s["throughput"] == pytest.approx(2.0)
    assert s["error_rate"] == pytest.approx(0.2)
    assert s["p50_ms"] == pytest.approx(2.5)


def row(rate, thr, p99, err=0.0):
    return {"rate": rate, "throughput": thr,
            "p99_ms": p99, "error_rate": err}


def test_capacity_stops_at_first_failure():
    rows = [row(100, 100, 10), row(200, 199, 20),
            row(400, 300, 90), row(800, 800, 5)]
    assert find_capacity(rows, 40.0, 0.001) == 200


def test_capacity_is_zero_when_lowest_rate_fails():
    assert find_capacity([row(50, 50, 99)], 40.0, 0.001) == 0.0


def test_slo_evaluation():
    metrics = {"capacity_rps": 350, "p99_ms_at_target": 55.0}
    limits = {"min_capacity_rps": 300,
              "max_p99_ms_at_target": 40.0,
              "max_gpu_memory_gib": 6.0}
    by = {r["gate"]: r for r in evaluate(metrics, limits)}
    assert by["min_capacity_rps"]["ok"]
    assert not by["max_p99_ms_at_target"]["ok"]
    assert not by["max_gpu_memory_gib"]["ok"]
