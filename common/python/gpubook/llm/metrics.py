from __future__ import annotations

import statistics
import time

import torch

GREEDY = dict(do_sample=False, temperature=None,
              top_p=None, top_k=None)


@torch.no_grad()
def quick_tps(model, tok, new_tokens=128,
              prompt_tokens=64, reps=3) -> float:
    """Median tokens per second at batch size one."""
    ids = torch.randint(
        1000, 5000, (1, prompt_tokens), device=model.device)
    mask = torch.ones_like(ids)
    rates = []
    for _ in range(reps + 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.generate(
            input_ids=ids, attention_mask=mask,
            max_new_tokens=new_tokens,
            min_new_tokens=new_tokens, **GREEDY)
        torch.cuda.synchronize()
        rates.append(new_tokens / (time.perf_counter() - t0))
    return statistics.median(rates[1:])

from gpubook.vision.latency import percentile


class TimingStreamer:
    """Record a timestamp for every generated step."""

    def __init__(self) -> None:
        self.stamps: list[float] = []
        self._seen_prompt = False
        self.end_time = 0.0

    def put(self, value) -> None:
        now = time.perf_counter()
        if not self._seen_prompt:
            self._seen_prompt = True  # first call is the prompt
            return
        self.stamps.append(now)

    def end(self) -> None:
        self.end_time = time.perf_counter()


def summarize_stamps(t0: float, stamps, batch: int = 1) -> dict:
    if not stamps:
        raise ValueError("no tokens were generated")
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    decode = stamps[-1] - stamps[0]
    return {
        "ttft_ms": (stamps[0] - t0) * 1e3,
        "itl_ms_mean": 1e3 * sum(gaps) / len(gaps) if gaps else 0.0,
        "itl_ms_p95": 1e3 * percentile(gaps, 0.95) if gaps else 0.0,
        "decode_tps_per_req": (len(gaps) / decode) if gaps else 0.0,
        "decode_tps_total": (len(gaps) / decode * batch) if gaps else 0.0,
        "e2e_ms": (stamps[-1] - t0) * 1e3,
        "tokens": len(stamps),
    }


def decode_ceiling(bandwidth_gbs: float, weights_gib: float) -> float:
    """Upper bound on single-stream decode, tokens per second.

    Every token reads all weights once."""
    return bandwidth_gbs * 1e9 / (weights_gib * 2**30)
