"""Small, unreduced objective results; callers own normalization and logging."""

from dataclasses import dataclass
from typing import Optional

from torch import Tensor


@dataclass(frozen=True)
class LossTerms:
    """A differentiable loss sum and its nonnegative token-weight sum.

    ``token_count`` can be zero for a padding-only evaluation batch. The caller
    decides whether to accumulate, normalize or reject that batch.
    Optional ``metrics`` are detached auxiliary sums with the same token
    denominator; they never contribute to backward.
    """

    loss_sum: Tensor
    token_count: Tensor
    metrics: Optional[dict[str, Tensor]] = None
