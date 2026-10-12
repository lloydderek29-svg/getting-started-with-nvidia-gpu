from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field

from gpubook.vision.latency import percentile


@dataclass
class LoadResult:
    latencies_ms: list[float] = field(default_factory=list)
    errors: int = 0
    dropped: int = 0
    offered: int = 0
    seconds: float = 0.0


async def closed_loop(call, concurrency, duration_s,
                      warmup_s=2.0) -> LoadResult:
    start = time.perf_counter()
    measure_from = start + warmup_s
    stop_at = measure_from + duration_s
    res = LoadResult(seconds=duration_s)

    async def worker():
        while time.perf_counter() < stop_at:
            t0 = time.perf_counter()
            try:
                await call()
            except Exception:
                if t0 >= measure_from:
                    res.errors += 1
                    res.offered += 1
                await asyncio.sleep(0.01)
                continue
            if t0 >= measure_from:
                ms = (time.perf_counter() - t0) * 1e3
                res.latencies_ms.append(ms)
                res.offered += 1

    await asyncio.gather(
        *(worker() for _ in range(concurrency)))
    return res


def summarize(res: LoadResult) -> dict:
    lat = res.latencies_ms
    out = {
        "offered": res.offered,
        "completed": len(lat),
        "errors": res.errors,
        "dropped": res.dropped,
        "throughput": (len(lat) / res.seconds
                       if res.seconds else 0.0),
        "error_rate": ((res.errors + res.dropped)
                       / res.offered if res.offered else 0.0),
    }
    if lat:
        out.update(
            p50_ms=percentile(lat, 0.50),
            p95_ms=percentile(lat, 0.95),
            p99_ms=percentile(lat, 0.99),
            max_ms=max(lat))
    else:
        inf = float("inf")
        out.update(p50_ms=inf, p95_ms=inf,
                   p99_ms=inf, max_ms=inf)
    return out

def poisson_schedule(rate: float, duration_s: float,
                     seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    t, out = 0.0, []
    while True:
        t += rng.expovariate(rate)
        if t >= duration_s:
            return out
        out.append(t)


async def open_loop(call, rate, duration_s, timeout_s=5.0,
                    max_inflight=5000, seed=0) -> LoadResult:
    schedule = poisson_schedule(rate, duration_s, seed)
    res = LoadResult(seconds=duration_s, offered=len(schedule))
    start = time.perf_counter()
    tasks: set = set()

    async def one(intended: float):
        try:
            await asyncio.wait_for(call(), timeout_s)
        except Exception:
            res.errors += 1
            return
        done = time.perf_counter()
        res.latencies_ms.append(
            (done - (start + intended)) * 1e3)

    for intended in schedule:
        wait = start + intended - time.perf_counter()
        if wait > 0:
            await asyncio.sleep(wait)
        if len(tasks) >= max_inflight:
            res.dropped += 1
            continue
        task = asyncio.create_task(one(intended))
        tasks.add(task)
        task.add_done_callback(tasks.discard)
    if tasks:
        await asyncio.gather(*list(tasks))
    return res


def find_capacity(rows: list[dict], p99_limit_ms: float,
                  max_error_rate: float) -> float:
    """Highest offered rate that meets the targets, with
    every lower rate meeting them too."""
    best = 0.0
    for row in sorted(rows, key=lambda r: r["rate"]):
        keeps_up = row["throughput"] >= 0.95 * row["rate"]
        ok = (row["p99_ms"] <= p99_limit_ms
              and row["error_rate"] <= max_error_rate
              and keeps_up)
        if not ok:
            break
        best = row["rate"]
    return best


async def step_load(call, rates, duration_s, seed=0):
    rows = []
    for rate in rates:
        res = await open_loop(call, rate, duration_s, seed=seed)
        row = summarize(res)
        row["rate"] = rate
        rows.append(row)
    return rows
