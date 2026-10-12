import argparse
import json
from pathlib import Path

import torch

from gpubook.llm.loading import fetch_model, inventory, load_model
from gpubook.llm.memory import (
    GIB, ModelShape, kv_cache_bytes, max_tokens)


@torch.no_grad()
def measure_cache(model, n_tokens: int, batch: int = 1):
    ids = torch.randint(
        1000, 5000, (batch, n_tokens), device=model.device)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = model.base_model(input_ids=ids, use_cache=True)
    cache = out.past_key_values
    del out
    torch.cuda.synchronize()
    held = torch.cuda.memory_allocated() - base
    peak = torch.cuda.max_memory_allocated() - base
    del cache
    return held, peak


def find_limit(model, cap: int) -> int:
    n, best = 1024, 0
    while n <= cap:
        try:
            measure_cache(model, n)
            best = n
            n *= 2
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            break
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--lengths", default="512,2048,8192")
    ap.add_argument("--find-limit", action="store_true")
    a = ap.parse_args()

    path = fetch_model(a.model)
    cfg = json.loads((Path(path) / "config.json").read_text())
    shape = ModelShape.from_config(
        cfg, inventory(path)["params"], a.model)
    model, _ = load_model(a.model)

    print(f"{'tokens':>8s} {'predicted':>11s} "
          f"{'measured':>11s} {'ratio':>6s} {'peak':>11s}")
    for n in (int(v) for v in a.lengths.split(",")):
        held, peak = measure_cache(model, n)
        pred = kv_cache_bytes(shape, n)
        print(f"{n:8d} {pred / 2**20:9.1f}Mi "
              f"{held / 2**20:9.1f}Mi {held / pred:6.2f} "
              f"{peak / 2**20:9.1f}Mi")

    if a.find_limit:
        cap = shape.max_positions or 1 << 20
        free, total = torch.cuda.mem_get_info()
        weights = model.get_memory_footprint() * 8 / shape.params / 8
        pred = max_tokens(
            shape, total / GIB, weights * 8)
        got = find_limit(model, cap)
        print(f"\ncache-only prediction : {pred} tokens")
        print(f"largest length that ran: {got} tokens")


if __name__ == "__main__":
    main()
