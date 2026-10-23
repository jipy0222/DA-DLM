"""Diffusion timestep embeddings and conditioning projection."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from .config import ConditioningMode


def timestep_embedding(
    timesteps: torch.Tensor,
    dimension: int,
    max_period: float = 10_000.0,
) -> torch.Tensor:
    if timesteps.ndim != 1:
        raise ValueError("timesteps must have shape [batch]")
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=timesteps.device, dtype=timesteps.dtype)
        / half
    )
    arguments = timesteps[:, None].float() * frequencies[None]
    embedding = torch.cat((torch.cos(arguments), torch.sin(arguments)), dim=-1)
    if dimension % 2:
        embedding = torch.cat(
            (embedding, torch.zeros_like(embedding[:, :1])), dim=-1
        )
    return embedding


class TimestepConditioner(nn.Module):
    def __init__(
        self,
        cond_dim: int,
        frequency_embedding_size: int = 256,
    ) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, cond_dim, bias=True),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim, bias=True),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        # Build timestep features in FP32, then run the conditioning MLP in
        # its parameter dtype with autocast disabled.
        with torch.autocast(device_type=timesteps.device.type, enabled=False):
            embedded = timestep_embedding(
                timesteps.float(), dimension=self.frequency_embedding_size
            )
            embedded = embedded.to(dtype=cast(nn.Linear, self.mlp[0]).weight.dtype)
            return cast('torch.Tensor', self.mlp(embedded))


def build_conditioning(
    conditioner: TimestepConditioner,
    mode: ConditioningMode,
    sigma: torch.Tensor,
) -> torch.Tensor:
    if mode is ConditioningMode.NONE:
        raise ValueError("ConditioningMode.NONE does not use a conditioner")
    if mode is ConditioningMode.ZERO_TIME_COMPAT:
        sigma = torch.zeros_like(sigma)
    elif mode is not ConditioningMode.ACTUAL_TIME:
        raise ValueError("unsupported conditioning mode: %s" % mode)
    return F.silu(conditioner(sigma))
