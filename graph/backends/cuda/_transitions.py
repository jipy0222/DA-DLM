"""Transition-layout helpers shared by compact CUDA graph kernels."""

from __future__ import annotations

import torch
from torch import Tensor


def dense_to_compact(transition_scores: Tensor) -> Tensor:
  """Convert absolute destination indices to compact forward offsets.

  Compact offset zero denotes ``source -> source + 1``. Entries whose
  destination would be outside the padded node dimension are canonical
  ``-inf``. The gather remains differentiable and is shared by training and
  no-grad kernels.
  """
  batch_size, node_count, _ = transition_scores.shape
  source = torch.arange(
    node_count, device=transition_scores.device).view(node_count, 1)
  offset = torch.arange(
    1, node_count, device=transition_scores.device).view(1, node_count - 1)
  destination = source + offset
  valid = destination < node_count
  safe_destination = destination.clamp_max(node_count - 1)
  compact = transition_scores.gather(
    2,
    safe_destination.unsqueeze(0).expand(batch_size, -1, -1),
  )
  return compact.masked_fill(~valid.unsqueeze(0), -torch.inf)


__all__ = ["dense_to_compact"]
