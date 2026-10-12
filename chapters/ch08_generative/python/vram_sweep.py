import argparse
import gc

import torch

from gpubook.genai.pipelines import load_pipeline
from gpubook.genai.vram import measure_generation

SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
CONFIGS = {
    "gpu resident": dict(offload="none"),
    "gpu + vae tiling": dict(offload="none", vae_tiling=True),
    "model offload": dict(offload="model", vae_tiling=True),
    "sequential offload": dict(
        offload="sequential", vae_tiling=True),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=SDXL)
    ap.add_argument("--sizes", default="768,1024,1536")
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--vae", default=None)
    a = ap.parse_args()
    sizes = [int(s) for s in a.sizes.split(",")]
    prompt = "a mountain village in winter, detailed"

    print(f"{'config':20s} {'size':>5s} {'sec':>7s} "
          f"{'peak GiB':>9s} {'note':>8s}")
    for name, opts in CONFIGS.items():
        pipe = load_pipeline(
            a.model, scheduler="dpmpp_2m", vae_id=a.vae, **opts)
        measure_generation(pipe, prompt, 512, 512, 4)  # warm-up
        for size in sizes:
            try:
                r = measure_generation(
                    pipe, prompt, size, size, a.steps)
                note = "BLACK" if r["broken"] else ""
                print(f"{name:20s} {size:5d} {r['seconds']:7.1f} "
                      f"{r['peak_gib']:9.2f} {note:>8s}")
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"{name:20s} {size:5d} "
                      f"{'out of memory':>26s}")
        del pipe
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
