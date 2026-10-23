"""Output heads kept outside the shared Transformer core."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class VocabularyHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        *,
        tied_weight: Optional[nn.Parameter] = None,
    ) -> None:
        super().__init__()
        if tied_weight is not None and tied_weight.shape != (vocab_size, hidden_size):
            raise ValueError("tied embedding weight has an incompatible shape")
        if tied_weight is None:
            self.weight = nn.Parameter(torch.zeros(vocab_size, hidden_size))
        else:
            self.weight = tied_weight
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.is_tied = tied_weight is not None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self.weight, self.bias)
