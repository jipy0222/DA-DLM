"""Mask probabilities, diffusion loss weights and sequence/block time sampling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor


@dataclass(frozen=True)
class MaskingSchedule:
    """Map times to positive reconstruction weights and masking probabilities.

    Times are in (0, 1). The default loglinear schedule returns (1/t, t).
    ``eps`` controls the probability floor in square/square_root and the
    endpoint smoothing in cosine; it does not shift loglinear probabilities.
    """

    name: str = "loglinear"
    eps: float = 1e-3

    def __post_init__(self) -> None:
        if self.name not in {"loglinear", "square", "square_root", "log", "cosine"}:
            raise ValueError("unknown masking schedule: %s" % self.name)
        if not 0 < self.eps < 1:
            raise ValueError("schedule eps must lie in (0, 1)")

    def __call__(self, t: Tensor) -> tuple[Tensor, Tensor]:
        if self.name == "loglinear":
            return t.reciprocal(), t
        if self.name in {"square", "square_root"}:
            exponent = 2.0 if self.name == "square" else 0.5
            probability = t.pow(exponent).clamp_min(self.eps)
            return exponent * t.pow(exponent - 1) / probability, probability
        if self.name == "log":
            log_two = t.new_tensor(2.0).log()
            probability = t.log1p() / log_two
            return cast('tuple[Tensor, Tensor]', (1 / (probability * log_two * (1 + t)), probability))
        angle = t * math.pi / 2
        probability = 1 - (1 - self.eps) * angle.cos()
        weight = (1 - self.eps) * angle.sin() / (probability + self.eps) * math.pi / 2
        return cast('tuple[Tensor, Tensor]', (weight, probability))


def sample_mdlm_time(batch_size: int, device: torch.device | str, *, antithetic: bool=True, t_min: float=0., t_max: float=1.,
                     generator: torch.Generator | None=None) -> Tensor:
    """Draw one time per sequence, optionally stratified across the local batch.

    Each rank samples its own batch. A zero RNG endpoint is moved to machine
    epsilon so inverse-time loss weights remain finite during validation.
    """
    if batch_size < 1 or not 0 <= t_min < t_max <= 1:
        raise ValueError("use a positive batch size and 0 <= t_min < t_max <= 1")
    times = torch.rand((batch_size, 1), device=device, generator=generator)
    if antithetic:
        offsets = torch.arange(batch_size, device=device)[:, None] / batch_size
        times = (times / batch_size + offsets) % 1
    times = times * (t_max - t_min) + t_min
    return times.clamp_min(torch.finfo(times.dtype).eps)


def sample_block_time(batch_size: int, sequence_length: int, block_size: int, device: torch.device | str, *,
                      antithetic: bool=True, t_min: float=0., t_max: float=1., generator: torch.Generator | None=None) -> Tensor:
    """Return [B, L] times shared within each target block.

    Stratification spans the local batch times its number of physical target
    blocks. A partial final block gets one time without extending the target.
    Fully padded blocks still occupy their sampling stratum; token weights
    exclude them from the objective.
    """
    if sequence_length < 1 or block_size < 1:
        raise ValueError("sequence_length and block_size must be positive")
    blocks = (sequence_length + block_size - 1) // block_size
    times = sample_mdlm_time(batch_size * blocks, device, antithetic=antithetic,
                             t_min=t_min, t_max=t_max, generator=generator)
    return times.reshape(batch_size, blocks).repeat_interleave(block_size, dim=1)[:, :sequence_length]
