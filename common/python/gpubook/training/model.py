from __future__ import annotations

import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torchvision.models import resnet18


class Checkpointed(nn.Module):
    """Recompute this block in the backward pass."""

    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.block = block

    def forward(self, x):
        return checkpoint(self.block, x, use_reentrant=False)


def resnet18_cifar(
    num_classes: int = 10, grad_checkpoint: bool = False
) -> nn.Module:
    m = resnet18(weights=None, num_classes=num_classes)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = nn.Identity()
    if grad_checkpoint:
        for name in ("layer1", "layer2", "layer3", "layer4"):
            setattr(m, name, Checkpointed(getattr(m, name)))
    return m
