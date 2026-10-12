import pytest
from PIL import Image

from gpubook.genai.workflow import (
    GenConfig, choose_offload, image_settings, load_gen_config,
    read_png_meta, save_png, settings_hash)


def test_hash_ignores_key_order_but_not_values():
    a = {"seed": 1, "steps": 20}
    b = {"steps": 20, "seed": 1}
    assert settings_hash(a) == settings_hash(b)
    assert settings_hash(a) != settings_hash({**a, "seed": 2})


def test_hash_changes_with_prompt():
    cfg = GenConfig()
    h1 = settings_hash(image_settings(cfg, "a cat", 1))
    h2 = settings_hash(image_settings(cfg, "a dog", 1))
    assert h1 != h2


def test_offload_thresholds():
    assert choose_offload(24.0, 6.5) == "none"
    assert choose_offload(9.0, 6.5) == "model"
    assert choose_offload(5.0, 6.5) == "sequential"


def test_png_metadata_round_trip(tmp_path):
    meta = {"seed": 7, "prompt": "x", "steps": 20}
    path = tmp_path / "a.png"
    save_png(Image.new("RGB", (8, 8)), path, meta)
    assert read_png_meta(path) == meta


def test_unknown_key_is_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("stepz: 10\n")
    with pytest.raises(ValueError):
        load_gen_config(p)
