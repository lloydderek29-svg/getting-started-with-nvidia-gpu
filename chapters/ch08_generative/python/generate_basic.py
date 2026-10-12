import argparse
import time

import torch

from gpubook.genai.pipelines import component_sizes, load_pipeline

SDXL = "stabilityai/stable-diffusion-xl-base-1.0"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--model", default=SDXL)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/ch08/basic.png")
    a = ap.parse_args()

    pipe = load_pipeline(a.model, scheduler="dpmpp_2m")
    for name, (n, b) in component_sizes(pipe).items():
        print(f"{name:16s} {n / 1e6:9.1f} M params "
              f"{b / 2**30:6.2f} GiB")

    gen = torch.Generator("cuda").manual_seed(a.seed)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    image = pipe(
        a.prompt, num_inference_steps=a.steps,
        guidance_scale=a.guidance, height=a.size,
        width=a.size, generator=gen).images[0]
    torch.cuda.synchronize()
    print(f"{time.perf_counter() - t0:.1f} s, "
          f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    image.save(a.out)


if __name__ == "__main__":
    main()
