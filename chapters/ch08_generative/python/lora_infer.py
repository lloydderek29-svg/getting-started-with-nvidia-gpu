import argparse

import torch

from gpubook.genai.grid import make_grid
from gpubook.genai.pipelines import load_pipeline

SD15 = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--model", default=SD15)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="results/ch08/lora_grid.png")
    a = ap.parse_args()

    pipe = load_pipeline(a.model, scheduler="dpmpp_2m")
    pipe.load_lora_weights(a.weights, adapter_name="custom")

    images, labels = [], []
    for strength in (0.0, 0.5, 1.0):
        pipe.set_adapters(["custom"], adapter_weights=[strength])
        gen = torch.Generator("cuda").manual_seed(a.seed)
        img = pipe(a.prompt, num_inference_steps=25,
                   height=a.size, width=a.size,
                   generator=gen).images[0]
        images.append(img)
        labels.append(f"LoRA strength {strength}")
    make_grid(images, labels).save(a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
