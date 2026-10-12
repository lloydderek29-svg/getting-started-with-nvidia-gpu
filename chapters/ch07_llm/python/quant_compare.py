import argparse
import gc
import time

import torch
from transformers import BitsAndBytesConfig

from gpubook.llm.loading import load_model
from gpubook.llm.metrics import quick_tps
from gpubook.llm.perplexity import perplexity, wikitext_text

MODES = {
    "bf16": (torch.bfloat16, None),
    "int8": (torch.float16,
             BitsAndBytesConfig(load_in_8bit=True)),
    "nf4": (torch.bfloat16, BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16)),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--max-tokens", type=int, default=40000)
    a = ap.parse_args()

    text = wikitext_text()
    print(f"{'mode':6s} {'load s':>7s} {'GiB':>6s} "
          f"{'ppl':>7s} {'tok/s':>7s}")
    for name, (dtype, quant) in MODES.items():
        t0 = time.perf_counter()
        model, tok = load_model(a.model, dtype, quant)
        load_s = time.perf_counter() - t0
        gib = model.get_memory_footprint() / 2**30
        ppl = perplexity(model, tok, text,
                         max_tokens=a.max_tokens)
        tps = quick_tps(model, tok)
        print(f"{name:6s} {load_s:7.1f} {gib:6.2f} "
              f"{ppl:7.3f} {tps:7.1f}")
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
