from __future__ import annotations

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(255.0**2 / mse))


class ClipScorer:
    def __init__(self, name="openai/clip-vit-base-patch32",
                 device="cuda"):
        self.device = device
        self.model = CLIPModel.from_pretrained(name).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(name)

    @staticmethod
    def _norm(x):
        return x / x.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def text_image(self, images, prompts):
        inputs = self.proc(
            text=prompts, images=images, return_tensors="pt",
            padding=True, truncation=True).to(self.device)
        img = self._norm(self.model.get_image_features(
            pixel_values=inputs["pixel_values"]))
        txt = self._norm(self.model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]))
        return (img * txt).sum(-1).cpu().tolist()

    @torch.no_grad()
    def image_image(self, a_images, b_images):
        def embed(images):
            px = self.proc(images=images, return_tensors="pt")
            px = px["pixel_values"].to(self.device)
            return self._norm(self.model.get_image_features(
                pixel_values=px))
        return (embed(a_images) * embed(b_images)).sum(-1).cpu().tolist()
# chapters/ch08_generative/tests/test_quality.py
import numpy as np
import pytest

from gpubook.genai.quality import psnr


def test_identical_images_have_infinite_psnr():
    a = np.full((4, 4, 3), 100, np.uint8)
    assert psnr(a, a) == float("inf")


def test_constant_offset_psnr():
    a = np.full((4, 4, 3), 100, np.uint8)
    b = a + 10
    assert psnr(a, b) == pytest.approx(28.13, abs=0.01)
