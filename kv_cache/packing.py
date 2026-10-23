"""Batch-wide indices for exact, right-padded KV cache packing.

Index preparation owns the data-dependent sizes. Each core call or commit
builds one request-local plan and shares it across layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class CachePackingPlan:
    lengths: torch.Tensor
    valid: torch.Tensor
    rows: torch.Tensor
    source_slots: torch.Tensor
    destination_slots: torch.Tensor

    @property
    def width(self) -> int:
        return self.valid.shape[1]

    def merge(self, past: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        """Pack [B,H,P,D] + [B,H,Q,D] into right-padded output rows.

        The plan's fixed-size integer indices also handle positions represented
        as [B,1,S,1] views.
        """

        # History and current data overwrite every valid slot; padding is
        # initialized after both writes.
        output = current.new_empty(
            current.shape[0], current.shape[1], self.width, current.shape[3]
        )
        past_width = min(past.shape[2], self.width)
        output[:, :, :past_width] = past[:, :, :past_width]
        output[self.rows, :, self.destination_slots] = current[
            self.rows, :, self.source_slots
        ]
        output.masked_fill_(~self.valid[:, None, :, None], 0)
        return output


def make_cache_packing_plan(
    history_lengths: torch.Tensor,
    current_valid: torch.Tensor,
    *,
    width: Optional[int] = None,
    _lengths: Optional[torch.Tensor] = None,
) -> CachePackingPlan:
    ranks = current_valid.long().cumsum(dim=-1) - 1
    # Reuse preflight totals when the caller already computed them for this mask.
    lengths = (
        history_lengths + current_valid.sum(dim=-1)
        if _lengths is None else _lengths
    )
    if width is None:
        width = int(lengths.max().item())
    valid = torch.arange(width, device=current_valid.device)[None] < lengths[:, None]
    # One batch-wide nonzero (a CUDA synchronization), shared by all layers.
    rows, source_slots = current_valid.nonzero(as_tuple=True)
    destination_slots = history_lengths[rows] + ranks[rows, source_slots]
    return CachePackingPlan(lengths, valid, rows, source_slots, destination_slots)
