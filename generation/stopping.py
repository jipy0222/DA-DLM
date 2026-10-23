"""Stop boundaries among accepted, generated tokens."""

from __future__ import annotations

import torch

from .config import StopConfig


def stop_boundaries(token_ids: torch.Tensor, eligible: torch.Tensor, stop: StopConfig) -> torch.Tensor:
    """Return the first eligible stop slot, or the row width if absent."""
    width = token_ids.shape[1]
    if stop.mode == "none":
        return torch.full((token_ids.shape[0],), width, device=token_ids.device,
                          dtype=torch.long)
    positions = torch.arange(width, device=token_ids.device)
    ids = token_ids.new_tensor(stop.token_ids)
    matches = torch.isin(token_ids, ids) & eligible
    return torch.where(matches, positions[None], width).amin(dim=-1)


def apply_block_stop(token_ids: torch.Tensor, eligible: torch.Tensor, stop: StopConfig, pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Deactivate a stopped suffix without filling any MASK to its left."""
    boundary = stop_boundaries(token_ids, eligible, stop)
    if stop.mode == "trim_suffix":
        slots = torch.arange(token_ids.shape[1], device=token_ids.device)
        token_ids = token_ids.masked_fill(slots[None] > boundary[:, None], pad_token_id)
    return token_ids, boundary
