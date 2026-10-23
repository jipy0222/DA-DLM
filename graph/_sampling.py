"""Internal inverse-CDF sampling shared by DAG references."""

from __future__ import annotations

import torch
from torch import Tensor


def _sample_categorical_from_uniforms(
  logits: Tensor,
  uniforms: Tensor,
  category_index: Tensor,
) -> Tensor:
  """Map one uniform per batch item to a categorical sample.

  Every row must contain at least one finite logit, and ``category_index`` is
  the shared one-dimensional index tensor for its columns. The first category
  whose CDF is strictly greater than the supplied uniform is selected. This
  skips leading zero-mass categories when the uniform is zero and clamps away
  a possible final-CDF rounding gap.
  """
  probabilities = torch.softmax(logits, dim=1)
  cumulative = probabilities.cumsum(dim=1)
  sampled = (cumulative <= uniforms.unsqueeze(1)).sum(dim=1)

  last_candidate = torch.where(
    torch.isfinite(logits),
    category_index.unsqueeze(0),
    torch.full_like(category_index.unsqueeze(0), -1),
  ).max(dim=1).values
  return torch.minimum(sampled, last_candidate)
