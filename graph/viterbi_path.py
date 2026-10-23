"""Pure PyTorch reference for maximum-score complete DAG paths."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

from .validation import mask_dense_dag, validate_scored_path_metadata


class ViterbiPath(NamedTuple):
  """A padded representation of a ragged batch of complete DAG paths.

  ``nodes[b, :lengths[b]]`` lists sample ``b``'s selected nodes in path
  order. Remaining entries are ``-1``. An unreachable sample has score
  ``-inf``, length zero, and an entirely ``-1`` node row.
  """

  scores: Tensor
  nodes: Tensor
  lengths: Tensor


@torch.no_grad()
def viterbi_path(
  node_scores: Tensor,
  transition_scores: Tensor,
  node_lengths: Tensor,
) -> ViterbiPath:
  """Return the maximum-score complete path through each DAG in a batch.

  A complete path starts at node 0 and ends at
  ``node_lengths[b] - 1``. Its score is the sum of every visited node score,
  including the start and terminal nodes, plus every traversed edge score.
  Only strict-forward edges inside each sample's valid node region
  participate.

  Args:
    node_scores: ``[B, N_max]`` additive scores for visiting nodes.
    transition_scores: Dense ``[B, N_max, N_max]`` additive edge scores or
      potentials. Entry ``[b, i, j]`` represents edge ``i -> j``.
    node_lengths: ``[B]`` long tensor with each sample's valid node count.

  Returns:
    A :class:`ViterbiPath` containing best scores, path-ordered node indices
    padded with ``-1``, and per-sample path lengths.

  The scores may be locally normalized log probabilities or unnormalized
  CRF potentials; no normalization or length correction is performed here.
  Exact ties use the lowest predecessor index, following ``torch.max``.
  Model-level policy decides which topology/transition combinations expose
  Viterbi decoding. This generation primitive is intentionally
  non-differentiable.
  """
  batch_size, max_node_count = validate_scored_path_metadata(
    node_scores,
    transition_scores,
    node_lengths,
  )
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

  # Backtrack all samples together. Strict-forward edges guarantee that every
  # reachable terminal arrives at node 0 in at most N_max visits.
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


__all__ = ["ViterbiPath", "viterbi_path"]
