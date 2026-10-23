"""ATen CUDA oracle implementations for custom-kernel parity tests.

These functions execute directly on CUDA tensors and share the backend-neutral
DP and inverse-CDF primitives that define the graph semantics.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

from ..._forward import (
  _all_paths_forward_log_weights,
  _fixed_length_alignment_forward_log_weights,
)
from ..._logspace import add_unless_impossible, stable_logsumexp
from ..._sampling import _sample_categorical_from_uniforms
from ...sample_alignment import SampledAlignment
from ...sample_path import SampledPath
from ...validation import (
  mask_dense_dag,
  validate_alignment_metadata,
  validate_sample_alignment_metadata,
  validate_sample_path_metadata,
  validate_scored_path_metadata,
  validate_transition_metadata,
)
from ...viterbi_path import ViterbiPath


def _require_cuda(*tensors: Tensor) -> None:
  if any(not tensor.is_cuda for tensor in tensors):
    raise ValueError("all CUDA graph inputs must be CUDA tensors")


def alignment_log_posterior_aten_cuda(
  emission_scores: Tensor,
  transition_scores: Tensor,
  node_lengths: Tensor,
  target_lengths: Tensor,
) -> Tensor:
  """Compute fixed-target alignment posteriors with CUDA ATen operations."""
  batch_size, max_target_length, max_node_count = validate_alignment_metadata(
    emission_scores,
    transition_scores,
    node_lengths,
    target_lengths,
  )
  _require_cuda(
    emission_scores,
    transition_scores,
    node_lengths,
    target_lengths,
  )
  device = emission_scores.device
  transitions, valid_nodes = mask_dense_dag(
    transition_scores,
    node_lengths,
  )

  target_index = torch.arange(max_target_length, device=device)
  valid_targets = target_index.unsqueeze(0) < target_lengths.unsqueeze(1)
  valid_states = valid_targets.unsqueeze(2) & valid_nodes.unsqueeze(1)
  emissions = emission_scores.masked_fill(~valid_states, -torch.inf)

  all_alpha = _fixed_length_alignment_forward_log_weights(
    emissions,
    transitions,
  )

  end_nodes = node_lengths - 1
  end_times = target_lengths - 1
  terminal_beta = emissions.new_full(
    (batch_size, max_node_count), -torch.inf)
  terminal_beta.scatter_(1, end_nodes.unsqueeze(1), 0)

  beta_next = emissions.new_full(
    (batch_size, max_node_count), -torch.inf)
  beta_steps_reversed = []
  for target_position in range(max_target_length - 1, -1, -1):
    at_end = cast(Tensor, target_position == end_times)
    before_end = target_position < end_times
    if target_position == max_target_length - 1:
      suffix = beta_next
    else:
      future_state = add_unless_impossible(
        emissions[:, target_position + 1, :],
        beta_next,
      )
      suffix = stable_logsumexp(
        transitions + future_state.unsqueeze(1),
        dim=2,
      ).squeeze(2)
    beta = torch.where(
      at_end.unsqueeze(1),
      terminal_beta,
      torch.where(
        before_end.unsqueeze(1),
        suffix,
        torch.full_like(suffix, -torch.inf),
      ),
    )
    beta_steps_reversed.append(beta)
    beta_next = beta
  all_beta = torch.stack(list(reversed(beta_steps_reversed)), dim=1)

  batch_index = torch.arange(batch_size, device=device)
  log_partition = all_alpha[batch_index, end_times, end_nodes]
  reachable_sample = ~torch.isneginf(log_partition)
  safe_partition = torch.where(
    reachable_sample,
    log_partition,
    torch.zeros_like(log_partition),
  )
  candidate = all_alpha + all_beta - safe_partition.view(-1, 1, 1)
  reachable_state = (
    valid_states
    & reachable_sample.view(-1, 1, 1)
    & ~torch.isneginf(all_alpha)
    & ~torch.isneginf(all_beta)
  )
  return torch.where(
    reachable_state,
    candidate,
    torch.full_like(candidate, -torch.inf),
  )


def path_log_partition_aten_cuda(
  transition_scores: Tensor,
  node_lengths: Tensor,
) -> Tensor:
  """Compute the all-path transition partition with CUDA ATen operations."""
  batch_size, _ = validate_transition_metadata(
    transition_scores,
    node_lengths,
  )
  _require_cuda(transition_scores, node_lengths)
  transitions, _ = mask_dense_dag(transition_scores, node_lengths)
  alpha = _all_paths_forward_log_weights(transitions, node_lengths)
  batch_index = torch.arange(batch_size, device=transition_scores.device)
  return alpha[batch_index, node_lengths - 1]


@torch.no_grad()
def viterbi_path_aten_cuda(
  node_scores: Tensor,
  transition_scores: Tensor,
  node_lengths: Tensor,
) -> ViterbiPath:
  """Compute maximum-score complete paths on CUDA tensors."""
  batch_size, max_node_count = validate_scored_path_metadata(
    node_scores,
    transition_scores,
    node_lengths,
  )
  _require_cuda(node_scores, transition_scores, node_lengths)
  transitions, valid_nodes = mask_dense_dag(
    transition_scores,
    node_lengths,
  )
  scores_at_nodes = node_scores.masked_fill(~valid_nodes, -torch.inf)

  best_scores = node_scores.new_full(
    (batch_size, max_node_count), -torch.inf)
  best_scores[:, 0] = scores_at_nodes[:, 0]
  backpointers = torch.full(
    (batch_size, max_node_count),
    -1,
    dtype=torch.long,
    device=node_scores.device,
  )

  for destination in range(1, max_node_count):
    candidates = (
      best_scores[:, :destination]
      + transitions[:, :destination, destination]
    )
    best_incoming, predecessor = candidates.max(dim=1)
    destination_score = scores_at_nodes[:, destination]
    reachable = (
      valid_nodes[:, destination]
      & ~torch.isneginf(best_incoming)
      & ~torch.isneginf(destination_score)
    )
    best_scores[:, destination] = torch.where(
      reachable,
      best_incoming + destination_score,
      torch.full_like(best_incoming, -torch.inf),
    )
    backpointers[:, destination] = torch.where(
      reachable,
      predecessor,
      torch.full_like(predecessor, -1),
    )

  batch_index = torch.arange(batch_size, device=node_scores.device)
  terminal_nodes = node_lengths - 1
  terminal_scores = best_scores[batch_index, terminal_nodes]
  reachable = ~torch.isneginf(terminal_scores)

  reversed_nodes = torch.full(
    (batch_size, max_node_count),
    -1,
    dtype=torch.long,
    device=node_scores.device,
  )
  current_nodes = terminal_nodes
  active = reachable
  for reverse_position in range(max_node_count):
    reversed_nodes[:, reverse_position] = torch.where(
      active,
      current_nodes,
      torch.full_like(current_nodes, -1),
    )
    at_start = active & current_nodes.eq(0)
    continue_backtracking = active & ~at_start
    predecessor = backpointers.gather(
      1, current_nodes.unsqueeze(1)).squeeze(1)
    current_nodes = torch.where(
      continue_backtracking,
      predecessor,
      current_nodes,
    )
    active = continue_backtracking & predecessor.ge(0)

  path_lengths = reversed_nodes.ge(0).sum(dim=1)
  path_positions = torch.arange(
    max_node_count, device=node_scores.device).unsqueeze(0)
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
  return ViterbiPath(
    scores=terminal_scores,
    nodes=forward_nodes,
    lengths=path_lengths,
  )


@torch.no_grad()
def sample_alignment_aten_cuda(
  emission_scores: Tensor,
  transition_scores: Tensor,
  node_lengths: Tensor,
  target_lengths: Tensor,
  uniforms: Tensor,
) -> SampledAlignment:
  """Sample fixed-target alignments with caller-provided CUDA uniforms."""
  batch_size, max_target_length, max_node_count = (
    validate_sample_alignment_metadata(
      emission_scores,
      transition_scores,
      node_lengths,
      target_lengths,
      uniforms,
    )
  )
  _require_cuda(
    emission_scores,
    transition_scores,
    node_lengths,
    target_lengths,
    uniforms,
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


@torch.no_grad()
def sample_path_aten_cuda(
  transition_scores: Tensor,
  node_lengths: Tensor,
  uniforms: Tensor,
) -> SampledPath:
  """Sample complete paths with caller-provided CUDA uniforms using ATen."""
  batch_size, max_node_count = validate_sample_path_metadata(
    transition_scores,
    node_lengths,
    uniforms,
  )
  _require_cuda(transition_scores, node_lengths, uniforms)
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


__all__ = [
  "alignment_log_posterior_aten_cuda",
  "path_log_partition_aten_cuda",
  "sample_alignment_aten_cuda",
  "sample_path_aten_cuda",
  "viterbi_path_aten_cuda",
]
