"""Shared forward DAG topology after callers have established node validity."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def dag_edges(valid: Tensor, anchors: Tensor, *, groups: Tensor | None,
              terminal: Tensor | None = None) -> Tensor:
    """Connect adjacent groups or free nodes up to their next right anchor.

    Masks have shape [rows, nodes]; optional group IDs have the same shape and
    already include the caller's BOS/EOS groups. Only differences between group
    IDs matter. A valid terminal connects directly to the physical endpoint.
    Callers own node/terminal validation and all training or generation budgets.
    """
    nodes = valid.shape[1]
    position = torch.arange(nodes, device=valid.device)
    if groups is not None:
        edges = groups[:, :, None].add(1).eq(groups[:, None, :])
    else:
        right_anchor = torch.where(anchors, position[None], nodes).flip(-1).cummin(-1).values.flip(-1)
        right_anchor = F.pad(right_anchor[:, 1:], (0, 1), value=nodes)
        edges = position[None, None, :] <= right_anchor[:, :, None]
    edges &= (position[:, None] < position[None, :])[None]
    edges &= valid[:, :, None] & valid[:, None, :]
    if terminal is not None:
        edges &= ~terminal[:, :, None] | position[None, None, :].eq(nodes - 1)
        edges[:, :, -1] |= terminal & valid
    return edges
