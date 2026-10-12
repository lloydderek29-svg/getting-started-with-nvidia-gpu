from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


class CaptionedImages(Dataset):
    def __init__(self, folder, resolution: int = 512) -> None:
        self.items = []
        for img in sorted(Path(folder).iterdir()):
            cap = img.with_suffix(".txt")
            if img.suffix.lower() in EXTS and cap.exists():
                self.items.append((img, cap.read_text().strip()))
        if not self.items:
            raise ValueError(f"no image and caption pairs in {folder}")
        self.tf = transforms.Compose([
            transforms.Resize(
                resolution,
                interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, caption = self.items[i]
        with Image.open(path) as im:
            return self.tf(im.convert("RGB")), caption

@torch.no_grad()
def cache_latents(ds, vae, tokenizer, text_encoder,
                  device, dtype, seed: int = 0):
    gen = torch.Generator(device=device).manual_seed(seed)
    latents, embeds = [], []
    for i in range(len(ds)):
        pixels, caption = ds[i]
        x = pixels[None].to(device, dtype)
        lat = vae.encode(x).latent_dist.sample(gen)
        latents.append((lat * vae.config.scaling_factor).cpu())
        ids = tokenizer(
            caption, padding="max_length", truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt").input_ids.to(device)
        embeds.append(text_encoder(ids)[0].cpu())
    return torch.cat(latents), torch.cat(embeds)
