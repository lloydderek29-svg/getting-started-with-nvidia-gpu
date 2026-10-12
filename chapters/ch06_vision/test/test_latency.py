import pytest

from gpubook.vision.latency import LatencyTracker, percentile


def test_percentile_interpolates():
    vals = [1.0, 2.0, 3.0, 4.0]
    assert percentile(vals, 0.0) == 1.0
    assert percentile(vals, 1.0) == 4.0
    assert percentile(vals, 0.5) == pytest.approx(2.5)


def test_tracker_skips_warmup_samples():
    t = LatencyTracker()
    for ms in (100.0, 10.0, 10.0, 10.0):
        t.add("detect", ms)
    s = t.summary(skip=1)["detect"]
    assert s["n"] == 3
    assert s["p50"] == 10.0