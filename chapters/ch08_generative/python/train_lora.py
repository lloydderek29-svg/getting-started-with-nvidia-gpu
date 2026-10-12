from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL, DDPMScheduler,
    StableDiffusionPipeline, UNet2DConditionModel)
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from transformers import CLIPTextModel, CLIPTokenizer

from gpubook.genai.lora_data import CaptionedImages, cache_latents

SD15 = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def save_lora(unet, out_dir: str) -> None:
    state = convert_state_dict_to_diffusers(
        get_peft_model_state_dict(unet))
    StableDiffusionPipeline.save_lora_weights(
        save_directory=out_dir, unet_lora_layers=state,
        safe_serialization=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=SD15)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/ch08/lora")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--dtype", default="bf16",
                    choices=["bf16", "fp16"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-every", type=int, default=200)
    ap.add_argument("--adam8bit", action="store_true")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = torch.device("cuda")
    dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float16

    tok = CLIPTokenizer.from_pretrained(a.model, subfolder="tokenizer")
    enc = CLIPTextModel.from_pretrained(
        a.model, subfolder="text_encoder").to(dev, dtype)
    vae = AutoencoderKL.from_pretrained(
        a.model, subfolder="vae").to(dev, dtype)
    sched = DDPMScheduler.from_pretrained(
        a.model, subfolder="scheduler")
    if sched.config.prediction_type != "epsilon":
        raise SystemExit("this script expects epsilon prediction")

    ds = CaptionedImages(a.data, a.resolution)
    latents, embeds = cache_latents(
        ds, vae, tok, enc, dev, dtype, a.seed)
    del vae, enc
    torch.cuda.empty_cache()
    print(f"{len(ds)} images cached, latents {tuple(latents.shape)}")

    unet = UNet2DConditionModel.from_pretrained(
        a.model, subfolder="unet").to(dev, dtype)
    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(
        r=a.rank, lora_alpha=a.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
    unet.enable_gradient_checkpointing()
    params = [p for p in unet.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    n_train = sum(p.numel() for p in params)
    n_all = sum(p.numel() for p in unet.parameters())
    print(f"trainable: {n_train / 1e6:.2f} M of {n_all / 1e6:.0f} M "
          f"({100 * n_train / n_all:.2f} %)")

    if a.adam8bit:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=a.lr)
    else:
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-2)
    warm = 50
    lr_sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm))
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(dtype == torch.float16))

    gen = torch.Generator().manual_seed(a.seed)
    n, T = len(latents), sched.config.num_train_timesteps
    torch.cuda.reset_peak_memory_stats()
    unet.train()
    t0, running = time.perf_counter(), None
    for step in range(1, a.steps + 1):
        for _ in range(a.accum):
            idx = torch.randint(0, n, (a.batch,), generator=gen)
            lat = latents[idx].to(dev, dtype)
            emb = embeds[idx].to(dev, dtype)
            noise = torch.randn_like(lat)
            t = torch.randint(0, T, (a.batch,), device=dev)
            noisy = sched.add_noise(lat, noise, t)
            with torch.autocast("cuda", dtype=dtype):
                pred = unet(noisy, t, emb).sample
            loss = F.mse_loss(pred.float(), noise.float())
            scaler.scale(loss / a.accum).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        lr_sched.step()
        opt.zero_grad(set_to_none=True)

        v = float(loss)
        running = v if running is None else 0.98 * running + 0.02 * v
        if step % 25 == 0:
            print(f"step {step:5d}  loss {running:.4f}  "
                  f"{time.perf_counter() - t0:6.0f}s")
        if step % a.ckpt_every == 0 and step < a.steps:
            save_lora(unet, f"{a.out}/step{step}")

    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"peak memory {peak:.2f} GiB")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    save_lora(unet, a.out)
    print(f"saved LoRA weights to {a.out}")


if __name__ == "__main__":
    main()
