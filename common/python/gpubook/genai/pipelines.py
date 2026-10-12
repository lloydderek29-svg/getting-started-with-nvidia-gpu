from __future__ import annotations

import torch
from diffusers import (
    AutoencoderKL,
    AutoPipelineForText2Image,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    UniPCMultistepScheduler,
    LCMScheduler

)

SCHEDULERS = {
    "euler": EulerDiscreteScheduler,
    "euler_a": EulerAncestralDiscreteScheduler,
    "dpmpp_2m": DPMSolverMultistepScheduler,
    "ddim": DDIMScheduler,
    "unipc": UniPCMultistepScheduler,
}

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16,
          "fp32": torch.float32}


def make_scheduler(pipe, name: str, **kwargs):
    return SCHEDULERS[name].from_config(
        pipe.scheduler.config, **kwargs)


def load_pipeline(model_id, dtype="fp16", scheduler=None,
                  offload="none", vae_tiling=False,
                  vae_id=None):
    kwargs = {"torch_dtype": DTYPES[dtype],
              "use_safetensors": True}
    if vae_id:
        kwargs["vae"] = AutoencoderKL.from_pretrained(
            vae_id, torch_dtype=DTYPES[dtype])
    try:
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id, variant="fp16", **kwargs)
    except (OSError, ValueError):
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id, **kwargs)

    if scheduler:
        extra = {"use_karras_sigmas": True} \
            if scheduler == "dpmpp_2m" else {}
        pipe.scheduler = make_scheduler(pipe, scheduler, **extra)

    if offload == "none":
        pipe.to("cuda")
    elif offload == "model":
        pipe.enable_model_cpu_offload()
    elif offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        raise ValueError(f"unknown offload mode {offload!r}")

    if vae_tiling:
        pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def component_sizes(pipe) -> dict:
    """Parameter count and bytes for every nn.Module part."""
    sizes = {}
    for name, part in pipe.components.items():
        if isinstance(part, torch.nn.Module):
            params = list(part.parameters())
            sizes[name] = (
                sum(p.numel() for p in params),
                sum(p.numel() * p.element_size() for p in params),
            )
    return sizes

def apply_speedups(pipe, channels_last=True, compile_unet=False):
    if channels_last:
        pipe.unet.to(memory_format=torch.channels_last)
    if compile_unet:
        pipe.unet = torch.compile(
            pipe.unet, mode="max-autotune", fullgraph=True)
    return pipe


def enable_lcm(pipe, adapter="latent-consistency/lcm-lora-sdxl"):
    pipe.scheduler = LCMScheduler.from_config(
        pipe.scheduler.config)
    pipe.load_lora_weights(adapter)
    return pipe
