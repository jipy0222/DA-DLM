"""Internal log-space primitives shared by PyTorch graph references."""

from __future__ import annotations

import torch
from torch import Tensor


def stable_logsumexp(values: Tensor, dim: int) -> Tensor:
  """Compute logsumexp with zero gradients for an all-``-inf`` slice."""
  maximum = values.max(dim=dim, keepdim=True).values
  all_impossible = maximum == -torch.inf

  maximum = maximum.detach()
  safe_maximum = maximum.masked_fill(all_impossible, 0)
  exp_sum = (values - safe_maximum).exp().sum(dim=dim, keepdim=True)
  safe_exp_sum = exp_sum.masked_fill(all_impossible, 1)
  return safe_exp_sum.log() + maximum


def add_unless_impossible(left: Tensor, right: Tensor) -> Tensor:
  """Add log scores while detaching states prohibited by ``-inf``."""
  possible = ~torch.isneginf(left) & ~torch.isneginf(right)
  candidate = left + right
  return torch.where(
    possible,
    candidate,
    torch.full_like(candidate, -torch.inf),
  )


__all__ = ["add_unless_impossible", "stable_logsumexp"]
