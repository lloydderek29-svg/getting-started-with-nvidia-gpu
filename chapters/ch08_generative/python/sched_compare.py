import argparse
import time

import torch
from PIL import Image, ImageDraw

from gpubook.genai.pipelines import (
    load_pipeline, make_scheduler)

SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
CASES = [("euler_a", 30), ("euler", 30), ("dpmpp_2m", 25),
         ("dpmpp_2m", 15), ("unipc", 12), ("ddim", 30)]


def make_grid(images, labels, cols=3):
    w, h = images[0].size
    rows = -(-len(images) // cols)
    sheet = Image.new("RGB", (cols * w, rows * h), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (img, label) in enumerate(zip(images, labels)):
        x, y = (i % cols) * w, (i // cols) * h
        sheet.paste(img, (x, y))
        draw.text((x + 8, y + 8), label, fill="white")
    return sheet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--model", default=SDXL)
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    pipe = load_pipeline(a.model)
    images, labels = [], []
    for name, steps in CASES:
        pipe.scheduler = make_scheduler(pipe, name)
        gen = torch.Generator("cuda").manual_seed(a.seed)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        img = pipe(a.prompt, num_inference_steps=steps,
                   height=a.size, width=a.size,
                   generator=gen).images[0]
        torch.cuda.synchronize()
        sec = time.perf_counter() - t0
        images.append(img)
        labels.append(f"{name} {steps} steps {sec:.1f}s")
        print(labels[-1])
    make_grid(images, labels).save("results/ch08/sched.png")


if __name__ == "__main__":
    main()
