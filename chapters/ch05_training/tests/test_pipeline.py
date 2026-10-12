import pytest
import torch
from gpubook.training.gpu_loader import (
    GpuTensorLoader, random_crop_flip)
from gpubook.training.config import load_config
from gpubook.training.repro import seed_everything


def test_unknown_config_key_is_rejected():
    with pytest.raises(ValueError):
        load_config(None, {"num_worker": 4})


def test_overrides_replace_defaults():
    cfg = load_config(None, {"batch_size": 64, "amp": None})
    assert cfg.batch_size == 64
    assert cfg.amp == "off"


def test_seeding_repeats_random_draws():
    seed_everything(123)
    a = torch.rand(4)
    seed_everything(123)
    b = torch.rand(4)
    assert torch.equal(a, b)

def test_crop_flip_keeps_shape_and_content():
    x = torch.arange(8.0).repeat(2, 3, 8, 1)  # columns 0..7
    out = random_crop_flip(x, 8, 8, pad=0)
    assert out.shape == x.shape
    for sample in out:
        row = sample[0, 0]
        fwd = torch.arange(8.0)
        assert torch.equal(row, fwd) or torch.equal(
            row, fwd.flip(0))


def test_loader_drops_last_in_training():
    imgs = torch.randint(0, 255, (10, 3, 8, 8),
                         dtype=torch.uint8)
    labels = torch.zeros(10, dtype=torch.long)
    loader = GpuTensorLoader(
        imgs, labels, 4, True, (0.5,) * 3, (0.5,) * 3)
    batches = list(loader)
    assert len(loader) == 2
    assert all(b[0].shape == (4, 3, 8, 8) for b in batches)
