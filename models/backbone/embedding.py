"""Token lookup with Kaiming-uniform embedding initialization."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dtype is not torch.long:
            raise TypeError("token_ids must be torch.long")
        return F.embedding(token_ids, self.weight)
