from __future__ import annotations

import math

import torch


def window_plan(n_tokens: int, max_len: int, stride: int):
    """Yield (begin, end, new_tokens) for each window."""
    prev_end = 0
    for begin in range(0, n_tokens, stride):
        end = min(begin + max_len, n_tokens)
        yield begin, end, end - prev_end
        prev_end = end
        if end == n_tokens:
            break


@torch.no_grad()
def perplexity(model, tok, text, max_len=2048,
               stride=1024, max_tokens=0) -> float:
    ids = tok(text, return_tensors="pt").input_ids
    if max_tokens:
        ids = ids[:, :max_tokens]
    nll, count = 0.0, 0
    for begin, end, new in window_plan(
            ids.size(1), max_len, stride):
        x = ids[:, begin:end].to(model.device)
        y = x.clone()
        y[:, :-new] = -100
        loss = model(x, labels=y).loss
        valid = int((y[:, 1:] != -100).sum())
        nll += float(loss) * valid
        count += valid
    return math.exp(nll / count)


def wikitext_text(split: str = "test") -> str:
    from datasets import load_dataset

    ds = load_dataset(
        "wikitext", "wikitext-2-raw-v1", split=split)
    return "\n\n".join(ds["text"])
