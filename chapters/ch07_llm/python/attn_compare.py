import argparse
import gc

import torch

from gpubook.bench import bench_cuda
from gpubook.llm.loading import load_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--tokens", type=int, default=4096)
    a = ap.parse_args()

    print(f"{'impl':20s} {'prefill ms':>11s} {'peak GiB':>9s}")
    for impl in ("eager", "sdpa", "flash_attention_2"):
        try:
            model, _ = load_model(a.model, attn=impl)
        except (ImportError, ValueError) as err:
            print(f"{impl:20s} unavailable "
                  f"({type(err).__name__})")
            continue
        ids = torch.randint(
            1000, 5000, (1, a.tokens), device=model.device)

        @torch.no_grad()
        def prefill():
            model.base_model(input_ids=ids, use_cache=False)

        torch.cuda.reset_peak_memory_stats()
        s = bench_cuda(prefill, warmup=3, repeats=10)
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"{impl:20s} {s.median_ms:11.1f} {peak:9.2f}")
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
