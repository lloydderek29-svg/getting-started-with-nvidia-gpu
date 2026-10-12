"""Run a matrix of training configs and print a summary."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from gpubook.training.report import aggregate, load_run

TRAIN = Path(__file__).with_name("train.py")
OUT = Path("results/ch05/matrix")
EPOCHS = 10
SEEDS = [0, 1, 2]

MATRIX = {
    "baseline": {},
    "workers8_pinned": dict(
        num_workers=8, pin_memory=True,
        persistent_workers=True),
    "gpu_loader": dict(loader="gpu"),
    "gpu_bf16": dict(loader="gpu", amp="bf16"),
    "gpu_bf16_cl": dict(
        loader="gpu", amp="bf16", channels_last=True),
    "gpu_bf16_cl_compile": dict(
        loader="gpu", amp="bf16", channels_last=True,
        compile="default"),
}


def flags(options: dict) -> list[str]:
    out = []
    for key, value in options.items():
        out += ["--" + key.replace("_", "-"), str(value)]
    return out


def run_one(name: str, options: dict, seed: int) -> Path:
    run_name = f"{name}_s{seed}"
    path = OUT / f"{run_name}.json"
    if path.exists():
        return path  # resume after an interruption
    cmd = [
        sys.executable, str(TRAIN),
        "--name", run_name, "--out-dir", str(OUT),
        "--epochs", str(EPOCHS), "--seed", str(seed),
    ] + flags(options)
    subprocess.run(cmd, check=True)
    return path


def fmt(pair, spec=".1f") -> str:
    mean, std = pair
    if mean is None:
        return "n/a"
    return f"{mean:{spec}} +/- {std:{spec}}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    table = {}
    for name, options in MATRIX.items():
        paths = [run_one(name, options, s) for s in SEEDS]
        table[name] = aggregate([load_run(p) for p in paths])

    base = table["baseline"]["img_s"][0]
    print(f"{'config':22s} {'img/s':>16s} {'x':>5s} "
          f"{'peak GiB':>9s} {'top1 %':>13s} "
          f"{'J/kimg':>8s} {'wait %':>7s}")
    for name, m in table.items():
        top1 = (m["top1"][0] * 100, m["top1"][1] * 100)
        print(f"{name:22s} {fmt(m['img_s'], '.0f'):>16s} "
              f"{m['img_s'][0] / base:5.2f} "
              f"{m['peak_gib'][0]:9.2f} "
              f"{fmt(top1, '.2f'):>13s} "
              f"{fmt(m['j_per_kimg']):>8s} "
              f"{m['data_wait_pct'][0]:7.1f}")


if __name__ == "__main__":
    main()
