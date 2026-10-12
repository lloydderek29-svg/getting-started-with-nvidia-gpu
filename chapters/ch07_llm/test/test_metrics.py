import pytest

from gpubook.llm.metrics import decode_ceiling, summarize_stamps


def test_summary_of_known_timestamps():
    s = summarize_stamps(0.5, [1.0, 1.1, 1.2, 1.3], batch=4)
    assert s["ttft_ms"] == pytest.approx(500.0)
    assert s["itl_ms_mean"] == pytest.approx(100.0)
    assert s["decode_tps_per_req"] == pytest.approx(10.0)
    assert s["decode_tps_total"] == pytest.approx(40.0)
    assert s["e2e_ms"] == pytest.approx(800.0)


def test_empty_generation_is_rejected():
    with pytest.raises(ValueError):
        summarize_stamps(0.0, [])


def test_decode_ceiling():
    assert decode_ceiling(1008.0, 16.0) == pytest.approx(
        58.67, abs=0.01)
