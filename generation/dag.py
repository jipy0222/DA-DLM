"""Inference-time DADLM node layout and legal transition edges."""

from __future__ import annotations

import torch
from torch import Tensor

from graph.layout import dag_edges


def initialize_dag(
    batch_size: int, *, block_size: int, dag_up_ratio: int, grouped: bool,
    content_lengths: Tensor, mask_token_id: int, pad_token_id: int,
    block_bos_id: int, block_eos_id: int, device: torch.device,
    start_token_id: int | None = None,
) -> Tensor:
    """Create fixed-width physical slots for a new graph block.

    ``content_lengths`` is a [batch] integer tensor on ``device`` counting
    active groups in grouped mode; it is ignored in free mode. Free graphs
    retain all physical candidates regardless of the remaining output budget.
    A supplied start is an already accepted ordinary content token at node 1.
    """
    if batch_size < 1 or block_size < 1:
        raise ValueError("batch_size and block_size must be positive")
    nodes = block_size * dag_up_ratio
    if nodes - 2 < block_size:
        raise ValueError("DAG content capacity must fit a full block")
    device = torch.device(device)
    if grouped:
        if content_lengths.shape != (batch_size,) or content_lengths.dtype not in (
            torch.int8, torch.int16, torch.int32, torch.int64,
        ):
            raise ValueError("content_lengths must be a [batch] integer tensor")
        if bool(((content_lengths < 0) | (content_lengths > block_size)).any()):
            raise ValueError("content_lengths must fit the active grouped block")
        if start_token_id is not None and bool((content_lengths == 0).any()):
            raise ValueError("a fixed start requires an active first group")
    ids = torch.full((batch_size, nodes), mask_token_id, dtype=torch.long, device=device)
    ids[:, 0] = block_bos_id
    ids[:, -1] = block_eos_id
    if grouped:
        position = torch.arange(nodes, device=device)
        inactive = ((position > 0) & (position < nodes - 1))[None] & (
            position[None] // dag_up_ratio >= content_lengths[:, None]
        )
        ids.masked_fill_(inactive, pad_token_id)
    if start_token_id is not None:
        ids[:, 1] = start_token_id
    return ids


def update_dag(
    ids: Tensor, *, dag_up_ratio: int, grouped: bool, content_lengths: Tensor,
    mask_token_id: int, pad_token_id: int, terminal_positions: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return copied IDs and legal edges after anchors and optional trim.

    ``terminal_positions[b]`` is an accepted content position, or physical N
    when there is no terminal. Trimming removes every content node to its
    right, including previously accepted anchors. The physical endpoint stays.
    Pass the same per-row ``content_lengths`` used to initialize this block.
    """
    if ids.ndim != 2 or ids.dtype != torch.long:
        raise ValueError("ids must be a [batch, nodes] long tensor")
    batch_size, nodes = ids.shape
    if dag_up_ratio < 2 or nodes < 2 or nodes % dag_up_ratio:
        raise ValueError("node count must be a multiple of dag_up_ratio >= 2")
    block_size = nodes // dag_up_ratio
    terminal = torch.as_tensor(terminal_positions, device=ids.device)
    if terminal.shape != (batch_size,) or terminal.dtype not in (
        torch.int8, torch.int16, torch.int32, torch.int64,
    ):
        raise ValueError("terminal_positions must be a [batch] integer tensor")
    terminal = terminal.to(torch.long)
    if bool(((terminal != nodes) & ((terminal < 1) | (terminal >= nodes - 1))).any()):
        raise ValueError("terminal positions must be content nodes or N")

    position = torch.arange(nodes, device=ids.device)
    content = (position > 0) & (position < nodes - 1)
    valid = ids.ne(pad_token_id)
    if grouped:
        valid &= ~content[None] | (
            position[None] // dag_up_ratio < content_lengths[:, None]
        )
    has_terminal = terminal.ne(nodes)
    any_terminal = bool(has_terminal.any())
    if any_terminal:
        row = torch.arange(batch_size, device=ids.device)[has_terminal]
        at = terminal[has_terminal]
        if bool((~valid[row, at] | ids[row, at].eq(mask_token_id)).any()):
            raise ValueError("terminal_positions must identify accepted content nodes")
        valid &= (position[None] <= terminal[:, None]) | position[None].eq(nodes - 1)

    if grouped:
        group = position // dag_up_ratio
        accepted = valid & content[None] & ids.ne(mask_token_id)
        counts = torch.zeros((batch_size, block_size), dtype=torch.long, device=ids.device)
        counts.scatter_add_(1, group[None].expand(batch_size, -1), accepted.long())
        if bool((counts > 1).any()):
            raise ValueError("grouped DAG has multiple accepted nodes in one group")
        anchor_position = torch.zeros_like(counts)
        anchor_position.scatter_add_(
            1, group[None].expand(batch_size, -1),
            torch.where(accepted, position[None], 0).expand(batch_size, -1),
        )
        group_accepted = counts.gather(1, group[None].expand(batch_size, -1)).bool()
        sibling = group_accepted & position[None].ne(
            anchor_position.gather(1, group[None].expand(batch_size, -1))
        )
        valid &= ~(content[None] & sibling)

    updated = ids.masked_fill(~valid, pad_token_id)
    groups = None
    if grouped:
        groups = (position // dag_up_ratio)[None].expand(batch_size, -1).clone()
        groups[:, 0] = -1
        groups[:, -1] = content_lengths
    terminal_mask = position[None].eq(terminal[:, None]) if any_terminal else None
    edges = dag_edges(valid, valid & updated.ne(mask_token_id), groups=groups, terminal=terminal_mask)
    return updated, edges
