import argparse

import numpy as np
import torch
from PIL import Image

from gpubook.genai.pipelines import load_pipeline
from gpubook.genai.workflow import read_png_meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("png")
    a = ap.parse_args()

    meta = read_png_meta(a.png)
    pipe = load_pipeline(
        meta["model"], dtype=meta["dtype"],
        scheduler=meta["scheduler"],
        offload=meta.get("offload", "none"))
    gen = torch.Generator("cuda").manual_seed(meta["seed"])
    img = pipe(
        meta["prompt"],
        negative_prompt=meta["negative_prompt"] or None,
        num_inference_steps=meta["steps"],
        guidance_scale=meta["guidance"], height=meta["height"],
        width=meta["width"], generator=gen).images[0]

    with Image.open(a.png) as orig:
        diff = np.abs(
            np.asarray(orig.convert("RGB"), dtype=np.int16)
            - np.asarray(img, dtype=np.int16))
    print(f"max pixel difference : {diff.max()}")
    print(f"identical            : {diff.max() == 0}")
    print(f"recorded environment : {meta['env']}")


if __name__ == "__main__":
    main()
