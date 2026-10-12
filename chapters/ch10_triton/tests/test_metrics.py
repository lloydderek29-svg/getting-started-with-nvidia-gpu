import pytest

from gpubook.serving.metrics import latency_breakdown


def snap(ok, ns_q, ns_in, ns_inf, ns_out, infer, execs):
    def stage(ns):
        return {"count": str(ok), "ns": str(ns)}
    return {
        "inference_count": str(infer),
        "execution_count": str(execs),
        "inference_stats": {
            "success": {"count": str(ok), "ns": "0"},
            "queue": stage(ns_q),
            "compute_input": stage(ns_in),
            "compute_infer": stage(ns_inf),
            "compute_output": stage(ns_out),
        },
    }


def test_breakdown_between_snapshots():
    before = snap(0, 0, 0, 0, 0, 0, 0)
    after = snap(100, 200_000_000, 20_000_000,
                 300_000_000, 10_000_000, 100, 20)
    bd = latency_breakdown(before, after)
    assert bd["queue_ms"] == pytest.approx(2.0)
    assert bd["compute_infer_ms"] == pytest.approx(3.0)
    assert bd["avg_batch"] == pytest.approx(5.0)
    assert bd["requests"] == 100
