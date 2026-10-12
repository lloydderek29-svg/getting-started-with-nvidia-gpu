import pytest
import torch
from PIL import Image

from gpubook.genai.lora_data import CaptionedImages


def test_dataset_pairs_images_with_captions(tmp_path):
    for name in ("a", "b"):
        Image.new("RGB", (80, 60), (200, 10, 10)).save(
            tmp_path / f"{name}.png")
        (tmp_path / f"{name}.txt").write_text(f"caption {name}")
    Image.new("RGB", (8, 8)).save(tmp_path / "orphan.png")

    ds = CaptionedImages(tmp_path, resolution=32)
    assert len(ds) == 2
    x, caption = ds[0]
    assert x.shape == (3, 32, 32)
    assert -1.0 <= float(x.min()) <= float(x.max()) <= 1.0
    assert caption == "caption a"


def test_empty_folder_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        CaptionedImages(tmp_path)
