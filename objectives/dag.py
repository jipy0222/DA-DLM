"""Fixed target blocks and explicit DADLM node/edge layouts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from graph.layout import dag_edges


@dataclass(frozen=True)
class DAGTargets:
    tokens: Tensor
    lengths: Tensor
    counts: Tensor
    fixed_bos: Tensor
    sample_weights: Tensor
    context_ids: Tensor
    batch_size: int
    block_count: int

    def expanded(self, block_bos: int, block_eos: int, pad_token_id: int) -> Tensor:
        """Bracket alignment targets with internal block BOS/EOS.

        ``block_eos`` follows each row's valid target tokens; a text EOS remains
        inside ``tokens``. The physical DAG endpoint is unchanged.
        """
        result = F.pad(self.tokens, (1, 1), value=pad_token_id)
        result[:, 0] = block_bos
        result.scatter_(1, (self.lengths + 1)[:, None], block_eos)
        return result


@dataclass(frozen=True)
class DAGLayout:
    ids: Tensor
    edge_valid: Tensor
    forced_nodes: Tensor


def prepare_dag_targets(input_ids: Tensor, *, block_size: int,
                        pad_token_id: int, bos_token_id: int,
                        source_length: int = 0, loss_mask: Tensor | None=None,
                        sample_weight: Tensor | None=None) -> DAGTargets:
    """Split a right-padded target suffix without changing the clean input.

    Counts exclude its initial BOS. Source and target storage padding are not
    prediction targets. A partial physical block is padded in both streams.
    Only the ordinary task mask (source/BOS/PAD excluded) is supported.
    """
    if input_ids.ndim != 2 or input_ids.dtype != torch.long:
        raise ValueError("input_ids must be a [batch, sequence] long tensor")
    if block_size < 1 or not 0 <= source_length < input_ids.shape[1]:
        raise ValueError("use a positive block_size and a nonempty target suffix")
    target = input_ids[:, source_length:]
    if not bool(target[:, 0].eq(bos_token_id).all()):
        raise ValueError("every target must start with the declared BOS token")
    valid = target.ne(pad_token_id)
    if bool((valid[:, 1:] & ~valid[:, :-1]).any()):
        raise ValueError("target PAD must form a suffix")
    scored = valid.clone()
    scored[:, 0] = False
    if loss_mask is not None:
        expected = F.pad(scored, (source_length, 0), value=False)
        if loss_mask.shape != input_ids.shape or not torch.equal(loss_mask, expected.to(loss_mask.dtype)):
            raise ValueError("loss_mask must score exactly target tokens except BOS and PAD")
    batch_size, target_width = target.shape
    weights = torch.ones(batch_size, device=input_ids.device, dtype=torch.float32)
    if sample_weight is not None:
        if sample_weight.shape != (batch_size,):
            raise ValueError("sample_weight must have one value per row")
        weights = sample_weight.to(device=input_ids.device, dtype=torch.float32)
        if not bool((torch.isfinite(weights) & (weights >= 0)).all()):
            raise ValueError("sample_weight must be finite and nonnegative")
    padding = (-target_width) % block_size
    context = F.pad(input_ids, (0, padding), value=pad_token_id) if padding else input_ids
    target = context[:, source_length:]
    block_count = target.shape[1] // block_size
    tokens = target.reshape(-1, block_size)
    counts = F.pad(scored, (0, padding), value=False).reshape(-1, block_size).sum(-1).float()
    fixed_bos = torch.zeros((batch_size, block_count), device=input_ids.device, dtype=torch.bool)
    fixed_bos[:, 0] = True
    return DAGTargets(tokens, tokens.ne(pad_token_id).sum(-1), counts,
                      fixed_bos.flatten(), weights.repeat_interleave(block_count),
                      context, batch_size, block_count)


def build_dag(targets: DAGTargets, noisy_tokens: Tensor, placements: Tensor, *,
              dag_up_ratio: int, grouped: bool, mask_token_id: int,
              pad_token_id: int, block_bos: int, block_eos: int,
              terminal_token_id: int | None=None) -> DAGLayout:
    """Inject visible targets and construct the final graph before link scoring.

    Grouped tails keep their original groups and bridge to the physical end.
    Free graphs retain MASK candidates, stopping only at a visible terminal.
    Blocks with no prediction targets have no active DAG queries or edges.
    """
    rows, block_size = targets.tokens.shape
    nodes = block_size * dag_up_ratio
    if nodes - 2 < block_size:
        raise ValueError("DAG content capacity must fit a full target block")
    if noisy_tokens.shape != targets.tokens.shape or placements.shape != targets.tokens.shape:
        raise ValueError("noisy tokens and placements must match target blocks")
    device = noisy_tokens.device
    row = torch.arange(rows, device=device)
    node = torch.arange(nodes, device=device)
    target_index = torch.arange(block_size, device=device)
    visible = (target_index[None] < targets.lengths[:, None]) & noisy_tokens.ne(mask_token_id)
    if bool((visible & ((placements < 1) | (placements >= nodes - 1))).any()):
        raise ValueError("visible targets require legal content placements")

    ids = torch.full((rows, nodes), mask_token_id, dtype=torch.long, device=device)
    ids[row[:, None].expand_as(visible)[visible], placements[visible]] = noisy_tokens[visible]
    ids[:, 0], ids[:, -1] = block_bos, block_eos
    forced = torch.full((rows, block_size + 2), -1, dtype=torch.long, device=device)
    forced[:, 0] = 0
    forced[:, 1:-1] = torch.where(visible, placements, -1)
    forced.scatter_(1, (targets.lengths + 1)[:, None], nodes - 1)

    content = (node > 0) & (node < nodes - 1)
    valid = torch.ones((rows, nodes), dtype=torch.bool, device=device)
    group = node // dag_up_ratio
    if grouped:
        valid = ~content[None] | (group[None] < targets.lengths[:, None])
        own = group[None].expand(rows, -1)
        sibling = visible.gather(1, own) & node[None].ne(placements.gather(1, own))
        valid &= ~(sibling & content[None])

    terminal = None
    if terminal_token_id is not None:
        terminal = ids.eq(terminal_token_id)
        terminal_position = torch.where(terminal, node[None], nodes).min(-1).values
        valid &= (node[None] <= terminal_position[:, None]) | (node[None] == nodes - 1)
    valid &= targets.counts.gt(0)[:, None]
    ids = ids.masked_fill(~valid, pad_token_id)
    groups = None
    if grouped:
        groups = (group + 1)[None].expand(rows, -1).clone()
        groups[:, 0] = 0
        groups[:, -1] = targets.lengths + 1
    edges = dag_edges(valid, valid & ids.ne(mask_token_id), groups=groups, terminal=terminal)
    return DAGLayout(ids, edges, forced)
