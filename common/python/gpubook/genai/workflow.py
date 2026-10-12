from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml
from PIL import Image
from PIL.PngImagePlugin import PngInfo


@dataclass
class GenConfig:
    name: str = "run"
    model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    scheduler: str = "dpmpp_2m"
    steps: int = 25
    guidance: float = 6.0
    width: int = 1024
    height: int = 1024
    seed: int = 1234
    images_per_prompt: int = 1
    negative_prompt: str = ""
    prompts: list[str] = field(default_factory=list)
    lora: str = ""
    lora_scale: float = 1.0
    offload: str = "auto"
    dtype: str = "fp16"
    out_dir: str = "outputs"


def load_gen_config(path) -> GenConfig:
    data = yaml.safe_load(Path(path).read_text()) or {}
    valid = {f.name for f in fields(GenConfig)}
    unknown = set(data) - valid
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return GenConfig(**data)


def image_settings(cfg: GenConfig, prompt: str, seed: int) -> dict:
    """Everything that determines one image."""
    return {
        "model": cfg.model, "scheduler": cfg.scheduler,
        "steps": cfg.steps, "guidance": cfg.guidance,
        "width": cfg.width, "height": cfg.height,
        "prompt": prompt, "negative_prompt": cfg.negative_prompt,
        "seed": seed, "lora": cfg.lora,
        "lora_scale": cfg.lora_scale, "dtype": cfg.dtype,
    }


def settings_hash(settings: dict) -> str:
    blob = json.dumps(settings, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:10]


def choose_offload(free_gib: float, weights_gib: float) -> str:
    """Starting heuristic for the offload mode."""
    if free_gib >= weights_gib * 2.2:
        return "none"
    if free_gib >= weights_gib * 1.1:
        return "model"
    return "sequential"


def save_png(image: Image.Image, path, meta: dict) -> None:
    info = PngInfo()
    info.add_text("gpubook", json.dumps(meta, sort_keys=True))
    image.save(path, pnginfo=info)


def read_png_meta(path) -> dict:
    with Image.open(path) as im:
        return json.loads(im.text["gpubook"])
