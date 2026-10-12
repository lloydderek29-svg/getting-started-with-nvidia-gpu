from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import torch

from gpubook.genai.pipelines import (
    component_sizes, load_pipeline)
from gpubook.genai.workflow import (
    choose_offload, image_settings, load_gen_config,
    save_png, settings_hash)


def versions() -> dict:
    out = {"torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0)}
    for pkg in ("diffusers", "transformers", "peft"):
        try:
            out[pkg] = version(pkg)
        except Exception:
            out[pkg] = "unknown"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = load_gen_config(a.config)

    run_dir = Path(cfg.out_dir) / cfg.name
    (run_dir / "images").mkdir(parents=True, exist_ok=True)
    shutil.copy(a.config, run_dir / "config.yaml")

    offload = cfg.offload
    free_gib = torch.cuda.mem_get_info()[0] / 2**30
    if offload == "auto":
        # Weights are unknown before loading; use a size
        # estimate by model family as a first guess.
        guess = 6.5 if "xl" in cfg.model.lower() else 2.5
        offload = choose_offload(free_gib, guess)
    print(f"free VRAM {free_gib:.1f} GiB, offload mode {offload}")

    pipe = load_pipeline(
        cfg.model, dtype=cfg.dtype, scheduler=cfg.scheduler,
        offload=offload, vae_tiling=offload != "none")
    if cfg.lora:
        pipe.load_lora_weights(cfg.lora, adapter_name="custom")
        pipe.set_adapters(
            ["custom"], adapter_weights=[cfg.lora_scale])
    total_gib = sum(
        b for _, b in component_sizes(pipe).values()) / 2**30
    env = versions()

    manifest = run_dir / "manifest.jsonl"
    made, skipped, t0 = 0, 0, time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    idx = 0
    for prompt in cfg.prompts:
        for k in range(cfg.images_per_prompt):
            seed = cfg.seed + idx
            settings = image_settings(cfg, prompt, seed)
            tag = settings_hash(settings)
            path = run_dir / "images" / f"{idx:03d}_{tag}.png"
            idx += 1
            if path.exists():
                skipped += 1
                continue
            gen = torch.Generator("cuda").manual_seed(seed)
            img = pipe(
                prompt, negative_prompt=cfg.negative_prompt or None,
                num_inference_steps=cfg.steps,
                guidance_scale=cfg.guidance, height=cfg.height,
                width=cfg.width, generator=gen).images[0]
            save_png(img, path, {**settings, "env": env,
                                 "offload": offload})
            with manifest.open("a") as f:
                f.write(json.dumps(
                    {"file": path.name, "hash": tag,
                     **settings}) + "\n")
            made += 1
            print(f"wrote {path.name}")

    elapsed = time.perf_counter() - t0
    summary = {
        "images": made, "skipped": skipped,
        "seconds": elapsed,
        "sec_per_image": elapsed / max(made, 1),
        "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
        "pipeline_weights_gib": total_gib,
        "env": env, "config": asdict(cfg),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"{made} new, {skipped} skipped, "
          f"{summary['sec_per_image']:.1f} s per image")


if __name__ == "__main__":
    main()
