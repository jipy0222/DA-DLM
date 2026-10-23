"""Layer normalization, AdaLN modulation, and gated residual operations."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BiaslessLayerNorm(nn.Module):
    """LayerNorm with FP32 statistics, a learned weight, and no learned bias."""

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = F.layer_norm(
            hidden_states.float(), (self.hidden_size,), eps=self.eps
        )
        return normalized * self.weight


def modulate(
    hidden_states: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    # Cast shift and scale to the hidden-state dtype before modulation.
    shift = shift.to(dtype=hidden_states.dtype)
    scale = scale.to(dtype=hidden_states.dtype)
    return hidden_states * (1 + scale) + shift


def gated_residual(
    residual: torch.Tensor,
    branch: torch.Tensor,
    gate: torch.Tensor | None,
) -> torch.Tensor:
    """Add a branch using the residual stream as the accumulator dtype."""

    branch = branch.to(dtype=residual.dtype)
    if gate is not None:
        branch = gate.to(dtype=residual.dtype) * branch
    return residual + branch


def zero_linear_(linear: nn.Linear) -> nn.Linear:
    with torch.no_grad():
        linear.weight.zero_()
        if linear.bias is not None:
            linear.bias.zero_()
    return linear
