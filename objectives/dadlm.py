"""DAG alignment emissions and unreduced local or CRF training objectives."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from graph import runtime
from graph.runtime import GraphBackend

from ._log_softmax_gather import log_softmax_gather
from .types import LossTerms


def _alignment_constraints(target_lengths: Tensor, forced_nodes: Tensor, node_valid: Tensor) -> tuple[Tensor, Tensor]:
    """Return endpoint-completed assignments and their structural support."""
    batch_size, target_count = forced_nodes.shape
    node_count = node_valid.shape[1]
    device = forced_nodes.device
    target_index = torch.arange(target_count, device=device)
    node_index = torch.arange(node_count, device=device)
    last_target = target_lengths - 1
    forced = forced_nodes.clone()
    forced[:, 0] = 0
    forced[torch.arange(batch_size, device=device), last_target] = node_count - 1
    valid_target = target_index[None, :] < target_lengths[:, None]
    interior = (target_index[None, :] > 0) & (target_index[None, :] < last_target[:, None])
    content_node = (node_index > 0) & (node_index < node_count - 1)
    free_allowed = interior[:, :, None] & content_node[None, None, :]
    forced_allowed = (forced[:, :, None].eq(node_index[None, None, :])
                      & (content_node[None, None, :] | ~interior[:, :, None]))
    allowed = torch.where(forced.ge(0)[:, :, None], forced_allowed, free_allowed)
    allowed &= valid_target[:, :, None] & node_valid[:, None, :]
    return forced, allowed


def build_emissions(
    logits: Tensor,
    targets: Tensor,
    target_lengths: Tensor,
    forced_nodes: Tensor,
    node_valid: Tensor,
    *,
    blocked_token_ids: Sequence[int]=(),
    partial_mask: Tensor | None = None,
    backend: GraphBackend = "auto",
) -> Tensor:
    """Match expanded targets to graph nodes with explicit hard constraints.

    ``targets`` includes both internal endpoints. A nonnegative forced node
    fixes that target's emission to zero at exactly one node. Endpoint target
    rows are always fixed to nodes 0 and N-1. Free interior rows use the
    allowed-vocabulary log probability of their target; they cannot match an
    endpoint or inactive node. Padded target rows have no valid emission.
    ``partial_mask`` removes finite soft scores only, leaving every hard
    prohibition at ``-inf``. CUDA fuses vocabulary normalization and gathering
    in a private buffer; ``backend='pytorch'`` keeps the reference path.
    """
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, node, vocabulary]")
    batch_size, node_count, _ = logits.shape
    if targets.ndim != 2 or targets.shape != forced_nodes.shape or targets.shape[0] != batch_size:
        raise ValueError("targets and forced_nodes must match [batch, target]")
    if target_lengths.shape != (batch_size,) or node_valid.shape != (batch_size, node_count):
        raise ValueError("target_lengths and node_valid must match the batch and node dimensions")
    if partial_mask is not None and partial_mask.shape != targets.shape:
        raise ValueError("partial_mask must match targets")
    if node_count < 2 or bool(((target_lengths < 2) | (target_lengths > targets.shape[1])).any()):
        raise ValueError("each expanded target must contain both endpoints")

    device = logits.device
    target_count = targets.shape[1]
    target_index = torch.arange(target_count, device=device)
    forced, allowed = _alignment_constraints(target_lengths, forced_nodes, node_valid)
    is_forced = forced.ge(0)
    valid_target = target_index[None, :] < target_lengths[:, None]

    with torch.autocast(device_type=device.type, enabled=False):
        # Forced and padded rows never read a vocabulary score. Zero is only
        # a safe gather index for them, not a replacement for a free target.
        target_ids = torch.where(valid_target & ~is_forced, targets, 0)
        soft = log_softmax_gather(logits, target_ids,
                                 blocked_token_ids=blocked_token_ids, backend=backend)
        if partial_mask is not None:
            soft = torch.where(partial_mask[:, :, None] & torch.isfinite(soft), 0.0, soft)

        return torch.where(is_forced[:, :, None], 0.0, soft).masked_fill(~allowed, -torch.inf)


def dadlm_objective(
    emission_scores: Tensor,
    transition_scores: Tensor,
    target_lengths: Tensor,
    token_counts: Tensor,
    *,
    loss_weights: Tensor,
    sample_weights: Tensor,
    crf: bool,
    backend: GraphBackend = "auto",
) -> LossTerms:
    """Return weighted DAG loss and scored-token count without normalization.

    A positive-count, positive-sample-weight row contributes ``-A`` in local
    mode or ``Z-A`` in CRF mode, where ``A`` fixes the expanded target and ``Z``
    sums every complete transition path. Both use the full physical node count.
    Noise loss weights affect the numerator only; sample weights affect both
    returned sums. Zero-scored rows never enter either dynamic program.
    """
    if emission_scores.ndim != 3 or transition_scores.ndim != 3:
        raise ValueError("emission_scores and transition_scores must be rank three")
    batch_size, _, node_count = emission_scores.shape
    if transition_scores.shape != (batch_size, node_count, node_count):
        raise ValueError("transition_scores must have shape [batch, node, node]")
    if any(value.shape != (batch_size,) for value in
           (target_lengths, token_counts, loss_weights, sample_weights)):
        raise ValueError("target lengths, counts, and weights must have shape [batch]")

    compute_dtype = (torch.float64 if emission_scores.dtype == torch.float64
                     or transition_scores.dtype == torch.float64 else torch.float32)
    with torch.autocast(device_type=emission_scores.device.type, enabled=False):
        emissions = emission_scores.to(compute_dtype)
        transitions = transition_scores.to(compute_dtype)
        row_weights = sample_weights.to(compute_dtype)
        count = (token_counts.to(compute_dtype) * row_weights).sum()
        active = (token_counts > 0) & (sample_weights > 0)
        if not bool(active.any()):
            # Empty reductions keep both score branches in autograd without
            # reading possibly nonfinite values from an unscored rank.
            zero = emissions[..., :0].sum() + transitions[..., :0].sum()
            return LossTerms(zero, count)

        selected = torch.nonzero(active, as_tuple=True)[0]
        node_lengths = torch.full_like(target_lengths[selected], node_count)
        alignment = runtime.alignment_log_partition(
            emissions[selected], transitions[selected], node_lengths,
            target_lengths[selected], backend=backend,
        )
        if crf:
            partition = runtime.path_log_partition(
                transitions[selected], node_lengths, backend=backend,
            )
            row_loss = partition - alignment
        else:
            row_loss = -alignment
        if not bool(torch.isfinite(row_loss).all()):
            bad_rows = selected[~torch.isfinite(row_loss)].tolist()
            raise ValueError(f"nonfinite DADLM objective for rows {bad_rows} (crf={crf})")
        weighted = row_loss * loss_weights[selected].to(compute_dtype) * row_weights[selected]
        return LossTerms(weighted.sum(), count)
