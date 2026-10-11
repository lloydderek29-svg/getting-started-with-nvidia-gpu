"""Estimate VRAM needs for inference and training."""
from __future__ import annotations

import argparse
from dataclasses import dataclass

GIB = 1024 ** 3


@dataclass(frozen=True)
class ModelSpec:
    name: str
    params_b: float      # parameters, in billions
    layers: int
    kv_heads: int
    head_dim: int


def weights_gib(spec: ModelSpec, bits: int) -> float:
    return spec.params_b * 1e9 * bits / 8 / GIB


def kv_cache_gib(spec: ModelSpec, tokens: int,
                 batch: int = 1, bytes_per: int = 2) -> float:
    per_token = (2 * spec.layers * spec.kv_heads
                 * spec.head_dim * bytes_per)
    return per_token * tokens * batch / GIB


def inference_gib(spec: ModelSpec, bits: int, tokens: int,
                  batch: int = 1,
                  overhead_gib: float = 1.5) -> float:
    return (weights_gib(spec, bits)
            + kv_cache_gib(spec, tokens, batch)
            + overhead_gib)


def full_training_gib(spec: ModelSpec) -> float:
    """Mixed precision Adam: about 16 bytes per parameter."""
    return spec.params_b * 1e9 * 16 / GIB


def report(spec: ModelSpec, tokens: int, batch: int,
           vram_gib: float | None) -> None:
    print(f"Model: {spec.name} ({spec.params_b:.1f}B params)")
    rows = [
        ("Inference fp16 ", inference_gib(spec, 16, tokens, batch)),
        ("Inference 8-bit", inference_gib(spec, 8, tokens, batch)),
        ("Inference 4-bit", inference_gib(spec, 4, tokens, batch)),
    ]
    for label, need in rows:
        line = f"  {label}: {need:6.1f} GiB"
        if vram_gib:
            ok = need <= 0.9 * vram_gib
            line += "  fits" if ok else "  does not fit"
        print(line)
    full = full_training_gib(spec)
    print(f"  Full fine-tune : {full:6.1f} GiB "
          "(before activations)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params-b", type=float, default=8.0)
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--vram", type=float, default=None)
    a = ap.parse_args()
    spec = ModelSpec("custom", a.params_b, a.layers,
                     a.kv_heads, a.head_dim)
    report(spec, a.tokens, a.batch, a.vram)


if __name__ == "__main__":
    main()
Running it for the 8B example against a 24 GB card gives output along these lines.
python chapters/ch01_architecture/python/vram_planner.py \
    --vram 24
Model: custom (8.0B params)
  Inference fp16 :   17.4 GiB  fits
  Inference 8-bit:   10.0 GiB  fits
  Inference 4-bit:    6.2 GiB  fits
  Full fine-tune :  119.2 GiB (before activations)
