import pytest

from gpubook.trt.gates import evaluate_gates


def test_max_and_min_rules():
    metrics = {"p99_ms_b1": 5.0, "top1_agreement": 0.99}
    gates = {"max_p99_ms_b1": 6.0, "min_top1_agreement": 0.995}
    by_name = {r["gate"]: r for r in evaluate_gates(metrics, gates)}
    assert by_name["max_p99_ms_b1"]["ok"]
    assert not by_name["min_top1_agreement"]["ok"]


def test_missing_metric_fails_the_gate():
    result = evaluate_gates({}, {"max_memory_gib": 2.0})[0]
    assert not result["ok"]
    assert result["note"] == "metric missing"


def test_unknown_gate_is_an_error():
    with pytest.raises(KeyError):
        evaluate_gates({}, {"max_nonsense": 1})
