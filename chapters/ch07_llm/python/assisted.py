import argparse
import time

import torch

from gpubook.llm.loading import load_model
from gpubook.llm.metrics import GREEDY


def timed(model, enc, new, **extra):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(
        **enc, max_new_tokens=new, **GREEDY, **extra)
    torch.cuda.synchronize()
    return out, new / (time.perf_counter() - t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target",
                    default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--draft",
                    default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--new", type=int, default=200)
    a = ap.parse_args()

    target, tok = load_model(a.target)
    draft, _ = load_model(a.draft)
    msgs = [{"role": "user", "content":
             "Explain how a GPU executes a warp."}]
    enc = tok.apply_chat_template(
        msgs, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    ).to(target.device)

    timed(target, enc, 16)  # warm-up
    base, base_tps = timed(target, enc, a.new)
    spec, spec_tps = timed(
        target, enc, a.new, assistant_model=draft)
    print(f"baseline   : {base_tps:6.1f} tok/s")
    print(f"speculative: {spec_tps:6.1f} tok/s")
    print(f"speedup    : {spec_tps / base_tps:6.2f}x")
    print(f"identical  : {torch.equal(base, spec)}")


if __name__ == "__main__":
    main()
