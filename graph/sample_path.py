"""Pure PyTorch reference for DAG path sampling with FFBS."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

from ._forward import _all_paths_forward_log_weights
from ._sampling import _sample_categorical_from_uniforms
from .validation import mask_dense_dag, validate_sample_path_metadata


class SampledPath(NamedTuple):
  """A padded representation of a ragged batch of sampled DAG paths."""

  nodes: Tensor
  lengths: Tensor


@torch.no_grad()
def sample_path(
  transition_scores: Tensor,
  node_lengths: Tensor,
  uniforms: Tensor,
) -> SampledPath:
  """Sample complete DAG paths by forward filtering/backward sampling.

  The sampled distribution assigns a complete path weight

  ``exp(sum_edge transition_score)``

  and normalizes over every strict-forward path from node 0 to each sample's
  terminal node. No emission score participates. ``uniforms[b, k]`` supplies
  the inverse-CDF draw for sample ``b`` at backward step ``k``; values must be
  in ``[0, 1)``. Random-number generation belongs to the caller.

  Args:
    transition_scores: Dense ``[B, N_max, N_max]`` additive edge scores or
      potentials. Only strict-forward edges within each sample participate.
    node_lengths: ``[B]`` long tensor with each sample's valid node count.
    uniforms: ``[B, N_max]`` pre-generated uniform draws. Entries beyond the
      maximum possible backward steps for a sample are ignored.

  Returns:
    A :class:`SampledPath`. ``nodes[b, :lengths[b]]`` lists the sampled path
    from node 0 to the sample terminal; remaining entries are ``-1``. An
    unreachable terminal produces length zero and an entirely ``-1`` row.

  If every visited node later emits from a normalized vocabulary distribution,
  exact marginalization of those token emissions contributes unit mass at each
  node. Thus this transition-only FFBS distribution is the path marginal of
  the joint path-and-token generation model; conditional token sampling belongs
  to the generation adapter.

  Locally normalized transition log probabilities and unnormalized CRF
  potentials are both accepted. This exact reference does not apply nucleus,
  top-p, length, or emission filtering.
  """
  batch_size, max_node_count = validate_sample_path_metadata(
    transition_scores,
    node_lengths,
    uniforms,
  )
  transitions, _ = mask_dense_dag(transition_scores, node_lengths)
  alpha = _all_paths_forward_log_weights(transitions, node_lengths)

  batch_index = torch.arange(batch_size, device=transition_scores.device)
  node_index = torch.arange(max_node_count, device=transition_scores.device)
  terminal_nodes = node_lengths - 1
  terminal_log_weights = alpha[batch_index, terminal_nodes]
  reachable = ~torch.isneginf(terminal_log_weights)

  reversed_nodes = torch.full(
    (batch_size, max_node_count),
    -1,
    dtype=torch.long,
    device=transition_scores.device,
  )
  current_nodes = terminal_nodes
  active = reachable
  fallback_logits = transition_scores.new_full(
    (batch_size, max_node_count), -torch.inf)
  fallback_logits[:, 0] = 0

  for reverse_step in range(max_node_count):
    reversed_nodes[:, reverse_step] = torch.where(
      active,
      current_nodes,
      torch.full_like(current_nodes, -1),
    )
    continue_sampling = active & current_nodes.ne(0)

    current_columns = current_nodes.view(batch_size, 1, 1).expand(
      -1, max_node_count, 1)
    incoming = transitions.gather(2, current_columns).squeeze(2)
    predecessor_logits = alpha + incoming
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
      uniforms[:, reverse_step],
      node_index,
    )
    current_nodes = torch.where(
      continue_sampling,
      sampled_predecessors,
      current_nodes,
    )
    active = continue_sampling

  path_lengths = reversed_nodes.ge(0).sum(dim=1)
  path_positions = node_index.unsqueeze(0)
  valid_path_positions = path_positions < path_lengths.unsqueeze(1)
  reversed_positions = (
    path_lengths.unsqueeze(1) - 1 - path_positions
  ).clamp_min(0)
  forward_nodes = reversed_nodes.gather(1, reversed_positions)
  forward_nodes = torch.where(
    valid_path_positions,
    forward_nodes,
    torch.full_like(forward_nodes, -1),
  )
  return SampledPath(nodes=forward_nodes, lengths=path_lengths)


__all__ = ["SampledPath", "sample_path"]
