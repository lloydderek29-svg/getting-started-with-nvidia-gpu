import numpy as np
import pytest

from gpubook.serving.client import (
    infer_with_retry, is_retryable, load_image_bytes)


class Flaky(Exception):
    def __init__(self, status):
        self._status = status

    def status(self):
        return self._status


def test_retryable_statuses():
    assert is_retryable("StatusCode.UNAVAILABLE")
    assert is_retryable("503")
    assert not is_retryable("400")
    assert not is_retryable(None)


def test_retries_then_succeeds():
    calls, sleeps = [], []

    def call():
        calls.append(1)
        if len(calls) < 3:
            raise Flaky("StatusCode.UNAVAILABLE")
        return "ok"

    out = infer_with_retry(call, sleep=sleeps.append,
                           rng=lambda: 0.5)
    assert out == "ok" and len(calls) == 3
    assert sleeps == [0.05, 0.1]


def test_bad_request_is_not_retried():
    calls = []

    def call():
        calls.append(1)
        raise Flaky("400")

    with pytest.raises(Flaky):
        infer_with_retry(call, sleep=lambda s: None)
    assert len(calls) == 1


def test_image_batch_shape_and_content(tmp_path):
    for i, blob in enumerate((b"aa", b"bbb")):
        (tmp_path / f"{i}.jpg").write_bytes(blob)
    arr = load_image_bytes(sorted(tmp_path.glob("*.jpg")))
    assert arr.shape == (2, 1) and arr.dtype == np.object_
    assert arr[1, 0] == b"bbb"
