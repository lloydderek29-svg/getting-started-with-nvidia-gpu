import argparse
import statistics
import time

import numpy as np
import torch

from gpubook.genai.grid import make_grid
from gpubook.genai.pipelines import (
    apply_speedups, enable_lcm, load_pipeline, make_scheduler)
from gpubook.genai.quality import ClipScorer, psnr
from gpubook.pareto import pareto_front

SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
PROMPTS = [
    "a lighthouse at dawn, oil painting",
    "a bowl of ramen on a wooden table, studio photo",
    "a futuristic city street in the rain, neon",
    "a portrait of an old fisherman, dramatic light",
]
# name, scheduler, steps, guidance
CONFIGS = [
    ("ref_dpmpp_50", "dpmpp_2m", 50, 6.0),
    ("euler_30", "euler", 30, 6.0),
    ("dpmpp_20", "dpmpp_2m", 20, 6.0),
    ("dpmpp_12", "dpmpp_2m", 12, 6.0),
    ("unipc_8", "unipc", 8, 6.0),
    ("lcm_4", "lcm", 4, 1.0),
]


def render(pipe, prompt, steps, guidance, size, seed):
    gen = torch.Generator("cuda").manual_seed(seed)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    img = pipe(prompt, num_inference_steps=steps,
               guidance_scale=guidance, height=size,
               width=size, generator=gen).images[0]
    torch.cuda.synchronize()
    return img, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=SDXL)
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--compile", action="store_true")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]

    pipe = load_pipeline(a.model, vae_id=None)
    apply_speedups(pipe, compile_unet=a.compile)
    clip = ClipScorer()
    render(pipe, PROMPTS[0], 4, 6.0, a.size, 0)  # warm-up

    images, times, mem = {}, {}, {}
    for name, sched, steps, guid in CONFIGS:
        if sched == "lcm":
            enable_lcm(pipe)
        else:
            pipe.scheduler = make_scheduler(pipe, sched)
        torch.cuda.reset_peak_memory_stats()
        ts = []
        for p in PROMPTS:
            for s in seeds:
                img, sec = render(pipe, p, steps, guid, a.size, s)
                images[(name, p, s)] = img
                ts.append(sec)
        times[name] = statistics.median(ts)
        mem[name] = torch.cuda.max_memory_allocated() / 2**30

    ref = CONFIGS[0][0]
    rows = []
    for name, *_ in CONFIGS:
        keys = [(p, s) for p in PROMPTS for s in seeds]
        mine = [images[(name, p, s)] for p, s in keys]
        base = [images[(ref, p, s)] for p, s in keys]
        text = clip.text_image(mine, [p for p, _ in keys])
        sim = clip.image_image(mine, base)
        pn = [psnr(np.asarray(m), np.asarray(b))
              for m, b in zip(mine, base)]
        finite = [v for v in pn if np.isfinite(v)]
        rows.append((name, times[name], mem[name],
                     statistics.mean(text), statistics.mean(sim),
                     statistics.mean(finite) if finite else
                     float("inf")))

    front = set(pareto_front([(r[0], r[1], r[4]) for r in rows]))
    print(f"{'config':14s} {'s/img':>7s} {'GiB':>6s} "
          f"{'clip txt':>9s} {'clip ref':>9s} {'psnr':>6s}  front")
    for n, t, m, ct, cr, ps in rows:
        print(f"{n:14s} {t:7.2f} {m:6.2f} {ct:9.3f} "
              f"{cr:9.3f} {ps:6.1f}  {'yes' if n in front else ''}")

    first = (PROMPTS[0], seeds[0])
    sheet = make_grid(
        [images[(n, *first)] for n, *_ in CONFIGS],
        [n for n, *_ in CONFIGS])
    sheet.save("results/ch08/bench_sheet.png")


if __name__ == "__main__":
    main()
