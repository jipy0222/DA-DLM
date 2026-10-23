"""Pure PyTorch reference for fixed-length Viterbi alignments."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

from .validation import mask_dense_dag, validate_alignment_metadata


class ViterbiAlignment(NamedTuple):
  """Best fixed-length scores and target-position-to-node alignments.

  ``nodes[b, t]`` is the DAG node aligned to target position ``t``. Target
  padding is ``-1``. An unreachable sample has score ``-inf`` and an entirely
  ``-1`` node row.
  """

  scores: Tensor
  nodes: Tensor


@torch.no_grad()
def viterbi_alignment(
  emission_scores: Tensor,
  transition_scores: Tensor,
  node_lengths: Tensor,
  target_lengths: Tensor,
) -> ViterbiAlignment:
  """Return the best exact-target-length alignment through each DAG.

  Candidate alignments start at node 0, end at
  ``node_lengths[b] - 1``, and visit exactly ``target_lengths[b]`` nodes.
  The score is the sum of the target-position/node emission scores and the
  traversed transition scores.

  Args:
    emission_scores: ``[B, T_max, N_max]`` additive score for aligning target
      position ``t`` to node ``j``.
    transition_scores: Dense ``[B, N_max, N_max]`` additive edge scores or
      potentials. Only strict-forward edges participate.
    node_lengths: ``[B]`` long tensor with each sample's valid node count.
    target_lengths: ``[B]`` long tensor with each sample's exact target
      length.

  Returns:
    A :class:`ViterbiAlignment` with best scores and a padded ``[B, T_max]``
    target-position-to-node alignment.

  ``-inf`` emission entries may impose forced-alignment constraints. Locally
  normalized transitions and unnormalized CRF potentials are both accepted;
  no normalization is performed. Exact ties use the lowest predecessor node
  index, following ``torch.max``. This assignment primitive is intentionally
  non-differentiable even when it is called from a training step.
  """
  batch_size, max_target_length, max_node_count = validate_alignment_metadata(
    emission_scores,
    transition_scores,
    node_lengths,
    target_lengths,
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

  best_scores = emission_scores.new_full(
    (batch_size, max_node_count), -torch.inf)
  best_scores[:, 0] = emissions[:, 0, 0]
  score_steps = [best_scores]
  backpointers = torch.full(
    (batch_size, max_target_length, max_node_count),
    -1,
    dtype=torch.long,
    device=emission_scores.device,
  )

  for target_position in range(1, max_target_length):
    candidates = best_scores.unsqueeze(2) + transitions
    best_incoming, predecessor = candidates.max(dim=1)
    current_emissions = emissions[:, target_position, :]
    reachable = (
      valid_targets[:, target_position].unsqueeze(1)
      & valid_nodes
      & ~torch.isneginf(best_incoming)
      & ~torch.isneginf(current_emissions)
    )
    best_scores = torch.where(
      reachable,
      best_incoming + current_emissions,
      torch.full_like(best_incoming, -torch.inf),
    )
    backpointers[:, target_position, :] = torch.where(
      reachable,
      predecessor,
      torch.full_like(predecessor, -1),
    )
    score_steps.append(best_scores)

  all_scores = torch.stack(score_steps, dim=1)
  batch_index = torch.arange(batch_size, device=emission_scores.device)
  terminal_nodes = node_lengths - 1
  terminal_targets = target_lengths - 1
  terminal_scores = all_scores[
    batch_index,
    terminal_targets,
    terminal_nodes,
  ]
  reachable = ~torch.isneginf(terminal_scores)

  alignment_nodes = torch.full(
    (batch_size, max_target_length),
    -1,
    dtype=torch.long,
    device=emission_scores.device,
  )
  current_nodes = terminal_nodes
  for reverse_offset in range(max_target_length):
    current_targets = target_lengths - 1 - reverse_offset
    active = reachable & current_targets.ge(0)
    safe_targets = current_targets.clamp_min(0)
    alignment_nodes[batch_index, safe_targets] = torch.where(
      active,
      current_nodes,
      alignment_nodes[batch_index, safe_targets],
    )
    predecessor = backpointers[
      batch_index,
      safe_targets,
      current_nodes,
    ]
    continue_backtracking = active & current_targets.gt(0)
    current_nodes = torch.where(
      continue_backtracking,
      predecessor,
      current_nodes,
    )

  return ViterbiAlignment(
    scores=terminal_scores,
    nodes=alignment_nodes,
  )


__all__ = ["ViterbiAlignment", "viterbi_alignment"]
