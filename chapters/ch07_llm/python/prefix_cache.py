import argparse
import copy
import time

import torch
from transformers import DynamicCache

from gpubook.llm.loading import load_model
from gpubook.llm.metrics import GREEDY

PREFIX = (
    "You are a careful assistant for a hardware company. "
    "Answer in one sentence. Policies: be accurate, "
    "cite specs, never guess. "
) * 60


@torch.no_grad()
def first_token_ms(model, ids, cache=None):
    kwargs = {}
    if cache is not None:
        kwargs["past_key_values"] = copy.deepcopy(cache)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.generate(
        input_ids=ids, attention_mask=torch.ones_like(ids),
        max_new_tokens=1, **GREEDY, **kwargs)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    default="Qwen/Qwen2.5-1.5B-Instruct")
    a = ap.parse_args()
    model, tok = load_model(a.model)

    prefix = tok(PREFIX, return_tensors="pt").input_ids
    prefix = prefix.to(model.device)
    cache = DynamicCache()
    with torch.no_grad():
        cache = model(
            input_ids=prefix, past_key_values=cache
        ).past_key_values
    print(f"prefix tokens: {prefix.size(1)}")

    question = tok(
        " What is the VRAM of the card?",
        return_tensors="pt", add_special_tokens=False,
    ).input_ids.to(model.device)
    full = torch.cat([prefix, question], dim=1)

    first_token_ms(model, full)  # warm-up
    cold = [first_token_ms(model, full) for _ in range(5)]
    warm = [first_token_ms(model, full, cache)
            for _ in range(5)]
    cold.sort()
    warm.sort()
    print(f"TTFT without cache: {cold[2]:7.1f} ms")
    print(f"TTFT with cache   : {warm[2]:7.1f} ms")


if __name__ == "__main__":
    main()
