import argparse
import time

import torch

from gpubook.llm.loading import load_model
from gpubook.llm.metrics import GREEDY


@torch.no_grad()
def tokens_per_second(model, batch, prompt, new):
    ids = torch.randint(
        1000, 5000, (batch, prompt), device=model.device)
    mask = torch.ones_like(ids)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.generate(
        input_ids=ids, attention_mask=mask,
        max_new_tokens=new, min_new_tokens=new, **GREEDY)
    torch.cuda.synchronize()
    return batch * new / (time.perf_counter() - t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--prompt", type=int, default=128)
    ap.add_argument("--new", type=int, default=128)
    a = ap.parse_args()

    model, _ = load_model(a.model)
    tokens_per_second(model, 1, a.prompt, 16)  # warm-up
    print(f"{'batch':>5s} {'total tok/s':>12s} "
          f"{'per request':>12s} {'peak GiB':>9s}")
    for batch in (1, 2, 4, 8, 16, 32, 64):
        torch.cuda.reset_peak_memory_stats()
        try:
            total = tokens_per_second(
                model, batch, a.prompt, a.new)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"{batch:5d} {'out of memory':>12s}")
            break
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"{batch:5d} {total:12.0f} "
              f"{total / batch:12.1f} {peak:9.2f}")


if __name__ == "__main__":
    main()
