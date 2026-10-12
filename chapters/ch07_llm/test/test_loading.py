import torch
from safetensors.torch import save_file

from gpubook.llm.loading import fits, inventory


def test_inventory_counts_params_and_bytes(tmp_path):
    save_file(
        {"a": torch.zeros(3, 4, dtype=torch.float16),
         "b": torch.zeros(5, dtype=torch.float32)},
        str(tmp_path / "model.safetensors"),
    )
    inv = inventory(tmp_path)
    assert inv["params"] == 17
    assert inv["bytes"] == 3 * 4 * 2 + 5 * 4
    assert inv["dtypes"] == {"F16": 12, "F32": 5}


def test_fits_respects_headroom_and_reserve():
    assert fits(10.0, 24.0)
    assert not fits(22.0, 24.0)
