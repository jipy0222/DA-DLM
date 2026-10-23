"""DADLM transition scores over an explicit graph topology."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import TransitionNormalization
from .types import TransitionOutput


def safe_logsumexp(hidden_states: torch.Tensor, dim: int) -> torch.Tensor:
    """Log-sum-exp with ``-inf`` outputs and zero gradients for all-``-inf`` slices."""

    maximum = hidden_states.max(dim=dim).values
    all_invalid = maximum.eq(-torch.inf)
    finite_maximum = maximum.masked_fill(all_invalid, 0.0)
    summed = (hidden_states - finite_maximum.unsqueeze(dim)).exp().sum(dim=dim)
    return summed.masked_fill(all_invalid, 1.0).log() + maximum.masked_fill(
        all_invalid, -torch.inf
    )


class DAGLink(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        nodes_per_block: int,
    ) -> None:
        super().__init__()
        if hidden_size % n_heads:
            raise ValueError("n_heads must divide hidden_size")
        if nodes_per_block <= 1:
            raise ValueError("nodes_per_block must be greater than one")
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.nodes_per_block = nodes_per_block
        self.query = nn.Linear(2 * hidden_size, hidden_size, bias=True)
        self.key = nn.Linear(2 * hidden_size, hidden_size, bias=True)
        self.gate = nn.Linear(2 * hidden_size, n_heads, bias=True)
        self.position_embedding = nn.Embedding(nodes_per_block, hidden_size)

    def forward(
        self,
        features: torch.Tensor,
        edge_valid: torch.Tensor,
        normalization: TransitionNormalization = TransitionNormalization.LOCAL,
    ) -> TransitionOutput:
        if features.ndim != 3 or features.shape[-1] != self.hidden_size:
            raise ValueError("features must have shape [batch, nodes, hidden]")
        batch, total_nodes, _ = features.shape
        if total_nodes % self.nodes_per_block:
            raise ValueError("node count must be divisible by nodes_per_block")
        block_count = total_nodes // self.nodes_per_block
        expected_edge_shape = (
            batch * block_count,
            self.nodes_per_block,
            self.nodes_per_block,
        )
        if edge_valid.shape != expected_edge_shape or edge_valid.dtype is not torch.bool:
            raise ValueError(
                "edge_valid must be boolean [batch*blocks, nodes_per_block, nodes_per_block]"
            )
        if edge_valid.device != features.device:
            raise ValueError("features and edge_valid must share a device")

        block_features = features.reshape(
            batch * block_count, self.nodes_per_block, self.hidden_size
        )
        position_ids = torch.arange(
            self.nodes_per_block, device=features.device
        )
        positional = self.position_embedding(position_ids)[None].expand(
            batch * block_count, -1, -1
        )
        augmented = torch.cat((block_features, positional), dim=-1)

        head_dim = self.hidden_size // self.n_heads
        # Keep the large projections in the caller's mixed-precision scope.
        query = self.query(augmented).reshape(
            batch * block_count, self.nodes_per_block, self.n_heads, head_dim
        )
        key = self.key(augmented).reshape(
            batch * block_count, self.nodes_per_block, self.n_heads, head_dim
        )
        # Score edges and gates in FP32. Functional parameter casts preserve
        # gradients when the module itself stores BF16 weights.
        with torch.autocast(device_type=features.device.type, enabled=False):
            per_head = torch.einsum(
                "bihd,bjhd->bijh", query.float(), key.float()
            ) / math.sqrt(head_dim)
            gate_logits = F.linear(
                augmented.float(), self.gate.weight.float(), self.gate.bias.float()
            )
            log_gate = F.log_softmax(gate_logits, dim=-1)
            per_head = per_head.masked_fill(~edge_valid[..., None], -torch.inf)

            if normalization is TransitionNormalization.LOCAL:
                has_edge = edge_valid.any(dim=-1)
                safe = per_head.masked_fill(~has_edge[:, :, None, None], 0.0)
                per_head = F.log_softmax(safe, dim=2)
                per_head = per_head.masked_fill(~edge_valid[..., None], -torch.inf)
            elif normalization is not TransitionNormalization.RAW_POTENTIAL:
                raise ValueError("unsupported transition normalization: %s" % normalization)

            scores = safe_logsumexp(
                per_head + log_gate[:, :, None, :], dim=-1
            )
            scores = scores.masked_fill(~edge_valid, -torch.inf)
        return TransitionOutput(scores, edge_valid, normalization)
