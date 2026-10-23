"""Pure PyTorch reference for fixed-target posterior alignment sampling."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

from ._forward import _fixed_length_alignment_forward_log_weights
from ._sampling import _sample_categorical_from_uniforms
from .validation import mask_dense_dag, validate_sample_alignment_metadata


class SampledAlignment(NamedTuple):
  """A target-position-to-node alignment padded with ``-1``."""

  nodes: Tensor


@torch.no_grad()
def sample_alignment(
  emission_scores: Tensor,
  transition_scores: Tensor,
  node_lengths: Tensor,
  target_lengths: Tensor,
  uniforms: Tensor,
) -> SampledAlignment:
  """Sample one fixed-target alignment from each posterior in a batch.

  An alignment starts at node 0, ends at ``node_lengths[b] - 1``, visits
  exactly ``target_lengths[b]`` nodes, and has unnormalized log weight

  ``sum_t emission_scores[b, t, a_t]``
  ``+ sum_t transition_scores[b, a_t, a_(t + 1)]``.

  ``uniforms[b, k]`` supplies the inverse-CDF draw at backward sampling step
  ``k``; step zero samples the predecessor of the fixed terminal node. Entries
  at and beyond ``target_lengths[b] - 1`` are ignored. Random-number generation
  belongs to the caller.

  Returns:
    A :class:`SampledAlignment` whose ``nodes`` tensor has shape
    ``[B, T_max]``. Entry ``nodes[b, t]`` is the node aligned to target
    position ``t``; target padding is ``-1``. An unreachable sample produces
    an entirely ``-1`` row without affecting other batch items.

  Padded states and non-forward edges are ignored. ``-inf`` emissions may
  impose alignment constraints. Locally normalized transitions and
  unnormalized CRF potentials are both accepted because sampling normalizes
  the fixed-target alignment posterior internally.
  """
  batch_size, max_target_length, max_node_count = (
    validate_sample_alignment_metadata(
      emission_scores,
      transition_scores,
      node_lengths,
      target_lengths,
      uniforms,
    )
  )
  transitions, valid_nodes = mask_dense_dag(
    transition_scores,
    node_lengths,
  )
  target_index = torch.arange(
    max_target_length, device=emission_scores.device)
  valid_targets = target_index.unsqueeze(0) < target_lengths.unsqueeze(1)
  emissions = emission_scores.masked_fill(
    ~(valid_targets.unsqueeze(2) & valid_nodes.unsqueeze(1)),
    -torch.inf,
  )
  alpha = _fixed_length_alignment_forward_log_weights(
    emissions,
    transitions,
  )

  batch_index = torch.arange(batch_size, device=emission_scores.device)
  node_index = torch.arange(max_node_count, device=emission_scores.device)
  terminal_targets = target_lengths - 1
  terminal_nodes = node_lengths - 1
  terminal_log_weights = alpha[
    batch_index,
    terminal_targets,
    terminal_nodes,
  ]
  reachable = ~torch.isneginf(terminal_log_weights)

  alignment_nodes = torch.full(
    (batch_size, max_target_length),
    -1,
    dtype=torch.long,
    device=emission_scores.device,
  )
  current_nodes = terminal_nodes
  fallback_logits = emission_scores.new_full(
    (batch_size, max_node_count), -torch.inf)
  fallback_logits[:, 0] = 0

  for reverse_offset in range(max_target_length):
    current_targets = terminal_targets - reverse_offset
    active = reachable & current_targets.ge(0)
    safe_current_targets = current_targets.clamp_min(0)
    alignment_nodes[batch_index, safe_current_targets] = torch.where(
      active,
      current_nodes,
      alignment_nodes[batch_index, safe_current_targets],
    )

    predecessor_targets = current_targets - 1
    continue_sampling = active & predecessor_targets.ge(0)
    safe_predecessor_targets = predecessor_targets.clamp_min(0)
    predecessor_alpha = alpha[
      batch_index,
      safe_predecessor_targets,
      :,
    ]
    current_columns = current_nodes.view(batch_size, 1, 1).expand(
      -1, max_node_count, 1)
    incoming = transitions.gather(2, current_columns).squeeze(2)
    predecessor_logits = predecessor_alpha + incoming
    predecessor_logits = predecessor_logits.masked_fill(
      node_index.unsqueeze(0) >= current_nodes.unsqueeze(1),
      -torch.inf,
    )
    sampling_logits = torch.where(
      continue_sampling.unsqueeze(1),
      predecessor_logits,
      fallback_logits,
    )
    sampled_predecessors = _sample_categorical_from_uniforms(
      sampling_logits,
      uniforms[:, reverse_offset],
      node_index,
    )
    current_nodes = torch.where(
      continue_sampling,
      sampled_predecessors,
      current_nodes,
    )

  return SampledAlignment(nodes=alignment_nodes)


__all__ = ["SampledAlignment", "sample_alignment"]
