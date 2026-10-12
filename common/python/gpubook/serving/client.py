from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np

RETRYABLE = ("UNAVAILABLE", "DEADLINE_EXCEEDED",
             "503", "504", "429")


def load_image_bytes(paths) -> np.ndarray:
    """Return an N x 1 object array of file contents."""
    data = [Path(p).read_bytes() for p in paths]
    arr = np.empty((len(data), 1), dtype=np.object_)
    for i, blob in enumerate(data):
        arr[i, 0] = blob
    return arr


def is_retryable(status) -> bool:
    return status is not None and any(
        code in str(status) for code in RETRYABLE)


def infer_with_retry(call, attempts=3, base_delay=0.05,
                     sleep=time.sleep, rng=random.random):
    for attempt in range(attempts):
        try:
            return call()
        except Exception as err:
            status = getattr(err, "status", lambda: None)()
            last = attempt == attempts - 1
            if last or not is_retryable(status):
                raise
            sleep(base_delay * 2 ** attempt * (0.5 + rng()))


def wait_ready(client, models, timeout=180.0, poll=1.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if client.is_server_ready() and all(
                    client.is_model_ready(m) for m in models):
                return True
        except Exception:
            pass
        time.sleep(poll)
    return False


def make_http_client(url="localhost:8000", concurrency=1,
                     timeout=60.0):
    import tritonclient.http as httpclient

    return httpclient.InferenceServerClient(
        url=url, concurrency=concurrency,
        connection_timeout=timeout, network_timeout=timeout)


def infer_http(client, model, in_name, array, out_names,
               binary=True, version=""):
    import tritonclient.http as httpclient
    from tritonclient.utils import np_to_triton_dtype

    inp = httpclient.InferInput(
        in_name, list(array.shape),
        np_to_triton_dtype(array.dtype))
    inp.set_data_from_numpy(array, binary_data=binary)
    outs = [httpclient.InferRequestedOutput(
        n, binary_data=binary) for n in out_names]
    res = client.infer(model, [inp], model_version=version,
                       outputs=outs)
    return [res.as_numpy(n) for n in out_names]


def infer_grpc(client, model, in_name, array, out_names,
               version="", timeout=None):
    import tritonclient.grpc as grpcclient
    from tritonclient.utils import np_to_triton_dtype

    inp = grpcclient.InferInput(
        in_name, list(array.shape),
        np_to_triton_dtype(array.dtype))
    inp.set_data_from_numpy(array)
    outs = [grpcclient.InferRequestedOutput(n)
            for n in out_names]
    res = client.infer(model, [inp], model_version=version,
                       outputs=outs, client_timeout=timeout)
    return [res.as_numpy(n) for n in out_names]
