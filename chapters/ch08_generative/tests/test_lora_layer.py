import torch
import torch.nn as nn

from gpubook.genai.lora_layer import LoRALinear, lora_params


def test_starts_identical_to_base():
    torch.manual_seed(0)
    base = nn.Linear(16, 8)
    layer = LoRALinear(base, r=4, alpha=4)
    x = torch.randn(5, 16)
    assert torch.allclose(layer(x), base(x))


def test_merged_layer_matches_unmerged():
    torch.manual_seed(0)
    layer = LoRALinear(nn.Linear(16, 8), r=4, alpha=8)
    with torch.no_grad():
        layer.B.normal_()
    x = torch.randn(5, 16)
    assert torch.allclose(layer(x), layer.merged()(x), atol=1e-5)


def test_only_adapter_parameters_train():
    layer = LoRALinear(nn.Linear(768, 768), r=8)
    trainable = sum(p.numel() for p in layer.parameters()
                    if p.requires_grad)
    assert trainable == lora_params(768, 768, 8) == 12288
    assert not layer.base.weight.requires_grad
