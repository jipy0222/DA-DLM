"""Optional, no-gradient DADLM alignment-coherence diagnostic.

The reference distribution ``q`` is the transition-weighted path law
conditioned on the *actual* expanded target length and on hard alignment
constraints.  It is not the all-length CRF path law.  This diagnostic is
defined for either locally normalized or raw CRF edge scores; those two
transition laws need not give the same value.
"""

from __future__ import annotations

import torch
from torch import Tensor

from graph import runtime
from graph.runtime import GraphBackend

from .dadlm import _alignment_constraints


@torch.no_grad()
def coherence_gap(
    emission_scores: Tensor,
    transition_scores: Tensor,
    target_lengths: Tensor,
    forced_nodes: Tensor,
    node_valid: Tensor,
    *,
    backend: GraphBackend = "auto",
) -> Tensor:
    """Return one signed coherence gap per already-active scored row.

    For each row, ``A`` sums complete paths weighted by transitions and
    emissions; ``Z_T`` sums the same structurally allowed paths with zero
    emissions.  With ``log_gamma[t, j]`` the log node posterior under ``Z_T``,
    the result is ``(A - Z_T) - sum_t logsumexp_j(log_gamma[t, j] + emission[t, j])``.
    The sum includes the two internal endpoint target positions, but excludes
    padded positions.  Forced ordinals, inactive nodes, and endpoint placement
    are structural constraints, even if a model score is numerically finite.
    """
    if emission_scores.ndim != 3 or transition_scores.ndim != 3:
        raise ValueError("emissions and transitions must be rank-three tensors")
    batch_size, target_count, node_count = emission_scores.shape
    if transition_scores.shape != (batch_size, node_count, node_count):
        raise ValueError("transition scores must have shape [batch, node, node]")
    if forced_nodes.shape != (batch_size, target_count):
        raise ValueError("forced_nodes must match emission target dimensions")
    if node_valid.shape != (batch_size, node_count):
        raise ValueError("node_valid must match emission node dimensions")
    if target_lengths.shape != (batch_size,):
        raise ValueError("target_lengths must have one value per row")
    if bool(((target_lengths < 2) | (target_lengths > target_count)
             | (target_lengths > node_count)).any()):
        raise ValueError("each expanded target length must fit both endpoints and nodes")

    _, allowed = _alignment_constraints(target_lengths, forced_nodes, node_valid)
    compute_dtype = (torch.float64 if torch.float64 in
                     (emission_scores.dtype, transition_scores.dtype)
                     else torch.float32)
    with torch.autocast(device_type=emission_scores.device.type, enabled=False):
        emissions = emission_scores.to(compute_dtype).masked_fill(~allowed, -torch.inf)
        valid_edges = node_valid[:, :, None] & node_valid[:, None, :]
        transitions = transition_scores.to(compute_dtype).masked_fill(
            ~valid_edges, -torch.inf)
        zero_emissions = torch.zeros_like(emissions).masked_fill(~allowed, -torch.inf)
        node_lengths = torch.full_like(target_lengths, node_count)

        aligned = runtime.alignment_log_partition(
            emissions, transitions, node_lengths, target_lengths, backend=backend)
        conditional_normalizer = runtime.alignment_log_partition(
            zero_emissions, transitions, node_lengths, target_lengths,
            backend=backend)
        log_gamma = runtime.alignment_log_posterior(
            zero_emissions, transitions, node_lengths, target_lengths,
            backend=backend)
        marginal = torch.logsumexp(log_gamma + emissions, dim=-1)
        target_active = (torch.arange(target_count, device=emissions.device)[None, :]
                         < target_lengths[:, None])
        marginal_sum = marginal.masked_fill(~target_active, 0).sum(dim=-1)
        result = aligned - conditional_normalizer - marginal_sum
        if not bool(torch.isfinite(result).all()):
            bad_rows = torch.nonzero(~torch.isfinite(result), as_tuple=True)[0].tolist()
            raise ValueError(f"nonfinite coherence gap for rows {bad_rows}")
        return result


__all__ = ["coherence_gap"]
