from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass
class TrainConfig:
    name: str = "baseline"
    data_dir: str = "data"
    out_dir: str = "results/ch05"
    loader: str = "dataloader"  # dataloader | gpu | synthetic
    epochs: int = 15
    batch_size: int = 256
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    clip_norm: float = 0.0
    seed: int = 0
    deterministic: bool = False
    num_workers: int = 4
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int = 2
    gpu_prefetch: bool = False
    amp: str = "off"  # off | fp16 | bf16
    accum_steps: int = 1
    channels_last: bool = False
    grad_checkpoint: bool = False
    compile: str = "off"  # off | default | reduce-overhead
    max_steps: int = 0  # 0 means full epochs


def load_config(path=None, overrides=None) -> TrainConfig:
    data: dict = {}
    if path:
        data = yaml.safe_load(Path(path).read_text()) or {}
    if overrides:
        data.update(
            {k: v for k, v in overrides.items() if v is not None}
        )
    valid = {f.name for f in fields(TrainConfig)}
    unknown = set(data) - valid
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    return TrainConfig(**data)


def _as_bool(text: str) -> bool:
    return text.lower() in ("1", "true", "yes", "on")


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None)
    for f in fields(TrainConfig):
        kind = type(f.default)
        parser.add_argument(
            "--" + f.name.replace("_", "-"),
            dest=f.name,
            type=_as_bool if kind is bool else kind,
            default=None,
        )
