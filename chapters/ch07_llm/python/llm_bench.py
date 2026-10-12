import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import BitsAndBytesConfig

from gpubook.env import snapshot
from gpubook.llm.metrics import (
    GREEDY, TimingStreamer, decode_ceiling, summarize_stamps)
from gpubook.llm.loading import load_model
from gpubook.monitor import GpuMonitor
from gpubook.vision.latency import percentile

QUANT = {
    "bf16": (torch.bfloat16, None),
    "int8": (torch.float16,
             BitsAndBytesConfig(load_in_8bit=True)),
    "nf4": (torch.bfloat16, BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16)),
}


@torch.no_grad()
def run_once(model, ids, mask, new_tokens):
    streamer = TimingStreamer()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.generate(
        input_ids=ids, attention_mask=mask,
        max_new_tokens=new_tokens, min_new_tokens=new_tokens,
        streamer=streamer, **GREEDY)
    return summarize_stamps(t0, streamer.stamps, ids.size(0))


def aggregate(runs: list[dict]) -> dict:
    out = {k: statistics.median(r[k] for r in runs)
           for k in runs[0]}
    ttft = [r["ttft_ms"] for r in runs]
    out["ttft_ms_p95"] = percentile(ttft, 0.95)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--quant", default="bf16",
                    choices=list(QUANT))
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--new-tokens", type=int, default=128)
    ap.add_argument("--batches", default="1,4,16")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--bandwidth-gbs", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dtype, quant = QUANT[a.quant]
    model, _ = load_model(a.model, dtype, quant)
    weights_gib = model.get_memory_footprint() / 2**30
    results = []

    for batch in (int(b) for b in a.batches.split(",")):
        ids = torch.randint(
            1000, 5000, (batch, a.prompt_tokens),
            device=model.device)
        mask = torch.ones_like(ids)
        try:
            for _ in range(a.warmup):
                run_once(model, ids, mask, a.new_tokens)
            torch.cuda.reset_peak_memory_stats()
            with GpuMonitor(interval=0.2) as mon:
                runs = [run_once(model, ids, mask, a.new_tokens)
                        for _ in range(a.repeats)]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            results.append({"batch": batch, "oom": True})
            continue

        row = aggregate(runs)
        row["batch"] = batch
        row["peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
        energy = mon.summary().get("energy_j")
        tokens = batch * a.new_tokens * a.repeats
        row["j_per_token"] = energy / tokens if energy else None
        results.append(row)

    print(f"{a.model}  {a.quant}  weights {weights_gib:.2f} GiB")
    print(f"{'batch':>5s} {'TTFT':>8s} {'ITL':>7s} "
          f"{'ITL p95':>8s} {'tok/s req':>10s} "
          f"{'tok/s all':>10s} {'GiB':>6s} {'J/tok':>7s}")
    for r in results:
        if r.get("oom"):
            print(f"{r['batch']:5d} out of memory")
            continue
        jt = r["j_per_token"]
        print(f"{r['batch']:5d} {r['ttft_ms']:7.0f}m "
              f"{r['itl_ms_mean']:6.1f}m "
              f"{r['itl_ms_p95']:7.1f}m "
              f"{r['decode_tps_per_req']:10.1f} "
              f"{r['decode_tps_total']:10.0f} "
              f"{r['peak_gib']:6.2f} "
              f"{jt if jt is None else round(jt, 3)!s:>7s}")

    if a.bandwidth_gbs and results and not results[0].get("oom"):
        ceil = decode_ceiling(a.bandwidth_gbs, weights_gib)
        got = results[0]["decode_tps_per_req"]
        print(f"\nsingle-stream ceiling {ceil:.0f} tok/s, "
              f"measured {got:.0f} ({100 * got / ceil:.0f} %)")

    if a.out:
        payload = {"env": snapshot(), "model": a.model,
                   "quant": a.quant, "args": vars(a),
                   "weights_gib": weights_gib,
                   "results": results}
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
