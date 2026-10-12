import pytest

from gpubook.serving.metrics import (
    gpu_summary, model_rates, parse_prometheus, per_model)


def exposition(ok, fail, req_us, infer, execs):
    return f'''# HELP nv_inference_request_success counter
# TYPE nv_inference_request_success counter
nv_inference_request_success{{model="pets",version="1"}} {ok}
nv_inference_request_failure{{model="pets",version="1"}} {fail}
nv_inference_request_duration_us{{model="pets",version="1"}} {req_us}
nv_inference_count{{model="pets",version="1"}} {infer}
nv_inference_exec_count{{model="pets",version="1"}} {execs}
nv_gpu_utilization{{gpu_uuid="GPU-a"}} 0.5
nv_gpu_utilization{{gpu_uuid="GPU-b"}} 0.3
nv_gpu_memory_used_bytes{{gpu_uuid="GPU-a"}} 1073741824
'''


def test_parser_reads_labels_and_values():
    s = parse_prometheus(exposition(10, 2, 5000, 10, 5))
    assert per_model(s, "nv_inference_request_failure") == {
        "pets": 2.0}
    assert gpu_summary(s)["util_pct"] == pytest.approx(40.0)
    assert gpu_summary(s)["mem_gib"] == pytest.approx(1.0)


def test_rates_between_scrapes():
    a = parse_prometheus(exposition(100, 0, 100_000, 100, 50))
    b = parse_prometheus(exposition(300, 4, 700_000, 500, 100))
    r = model_rates(a, b, 10.0)["pets"]
    assert r["rps"] == pytest.approx(20.0)
    assert r["fail_rps"] == pytest.approx(0.4)
    assert r["avg_batch"] == pytest.approx(8.0)
    assert r["latency_ms"] == pytest.approx(3.0)


def test_counter_reset_is_not_negative():
    a = parse_prometheus(exposition(500, 0, 9_000_000, 500, 50))
    b = parse_prometheus(exposition(10, 0, 20_000, 10, 5))
    assert model_rates(a, b, 1.0)["pets"]["rps"] == 0.0
