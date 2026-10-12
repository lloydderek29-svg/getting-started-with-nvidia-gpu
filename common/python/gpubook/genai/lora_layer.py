from __future__ import annotations

import math

import torch
import torch.nn as nn


def lora_params(d_in: int, d_out: int, r: int) -> int:
    return r * (d_in + d_out)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8,
                 alpha: float = 8.0) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.scale = alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = (x @ self.A.T) @ self.B.T
        return self.base(x) + delta * self.scale

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        """Fold the update into a plain linear layer."""
        out = nn.Linear(self.base.in_features,
                        self.base.out_features,
                        bias=self.base.bias is not None)
        out.weight.copy_(
            self.base.weight + self.scale * (self.B @ self.A))
        if self.base.bias is not None:
            out.bias.copy_(self.base.bias)
        return out
