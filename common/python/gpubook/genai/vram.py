from __future__ import annotations

import time

import numpy as np
import torch


def latent_tokens(height: int, width: int,
                  factor: int = 8) -> int:
    return (height // factor) * (width // factor)


def attention_matrix_gib(tokens: int, heads: int = 8,
                         bytes_per: int = 2) -> float:
    """Size of one naive attention score matrix per image."""
    return tokens * tokens * heads * bytes_per / 2**30


def looks_broken(image) -> bool:
    """Detect all-black output, a symptom of NaN values."""
    arr = np.asarray(image, dtype=np.float32)
    return bool(arr.mean() < 1.0 or np.isnan(arr).any())


def measure_generation(pipe, prompt, height, width, steps=20,
                       batch=1, seed=0) -> dict:
    gen = torch.Generator("cuda").manual_seed(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = pipe(prompt, num_inference_steps=steps, height=height,
               width=width, num_images_per_prompt=batch,
               generator=gen)
    torch.cuda.synchronize()
    sec = time.perf_counter() - t0
    return {
        "seconds": sec,
        "images_per_s": batch / sec,
        "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
        "broken": any(looks_broken(i) for i in out.images),
    }
