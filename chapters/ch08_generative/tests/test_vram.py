import numpy as np
import pytest
from PIL import Image

from gpubook.genai.vram import (
    attention_matrix_gib, latent_tokens, looks_broken)


def test_latent_tokens_scale_with_pixels():
    assert latent_tokens(512, 512) == 4096
    assert latent_tokens(1024, 1024) == 4 * 4096


def test_attention_matrix_grows_sixteenfold_per_doubling():
    small = attention_matrix_gib(latent_tokens(512, 512))
    large = attention_matrix_gib(latent_tokens(1024, 1024))
    assert large == pytest.approx(16 * small)
    assert small == pytest.approx(0.03125, rel=1e-6)
    assert large == pytest.approx(0.5, rel=1e-6)


def test_black_image_is_flagged():
    black = Image.fromarray(np.zeros((8, 8, 3), np.uint8))
    gray = Image.fromarray(np.full((8, 8, 3), 128, np.uint8))
    assert looks_broken(black)
    assert not looks_broken(gray)
