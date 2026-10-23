"""Pure PyTorch reference for fixed-target DAG alignment partitions.

This module implements an explicit dynamic program and serves as a numerical
reference for the graph backends.

Only dense absolute-destination transitions are accepted here.

The normal call validates metadata and lengths only.  Valid score entries must
be finite or ``-inf``; use ``strict_validate_alignment_inputs`` when a test or
debug run should scan complete tensors.  The dynamic program always masks
padding, diagonal, and backward entries, while strict validation additionally
requires those non-edge transition entries to use canonical ``-inf`` storage.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ._forward import _fixed_length_alignment_forward_log_weights
from .validation import mask_dense_dag, validate_alignment_metadata


def alignment_log_partition(
  emission_scores: Tensor,
  transition_scores: Tensor,
  node_lengths: Tensor,
  target_lengths: Tensor,
) -> Tensor:
  """Sum all DAG alignments with a prescribed target length in log space.

  Args:
    emission_scores: ``[B, T_max, N_max]`` tensor. Entry ``[b, t, j]`` is
      the log emission score for target position ``t`` at DAG node ``j``.
    transition_scores: Dense ``[B, N_max, N_max]`` tensor. Entry
      ``[b, i, j]`` is the log score of edge ``i -> j``. Nodes must be in
      strict topological order. Only forward entries participate; canonical
      inputs represent diagonal, backward, and prohibited edges with ``-inf``.
    node_lengths: ``[B]`` long tensor containing each sample's valid number
      of DAG nodes. The fixed terminal node is ``node_lengths[b] - 1``.
    target_lengths: ``[B]`` long tensor containing each sample's prescribed
      number of emitted target positions. A valid path visits exactly this
      many nodes and therefore contains ``target_lengths[b] - 1`` edges.

  Returns:
    A ``[B]`` tensor containing the log-sum-exp of every valid path that
    starts at node 0, ends at the sample's terminal node, and emits every
    target position in order. No negation, reduction, loss scaling, or CRF
    all-length normalization is applied.

  ``-inf`` can be used to prohibit either an emission alignment or an edge.
  Padded target/node regions are ignored. Transition scores may be locally
  normalized log probabilities or unnormalized CRF potentials; this function
  does not normalize or otherwise reinterpret them. The correctness contract
  covers float32 and float64; mixed and lower-precision score tensors must be
  explicitly converted by the caller.
  """
  batch_size, max_target_length, max_node_count = validate_alignment_metadata(
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

  # Mask padded emissions to prevent invalid padded values from entering
  # intermediate states in a heterogeneous batch.
  emissions = emission_scores.masked_fill(
    ~(valid_targets.unsqueeze(2) & valid_nodes.unsqueeze(1)),
    -torch.inf,
  )
  all_alpha = _fixed_length_alignment_forward_log_weights(
    emissions,
    transitions,
  )
  batch_index = torch.arange(batch_size, device=device)
  return all_alpha[
    batch_index,
    target_lengths - 1,
    node_lengths - 1,
  ]


__all__ = ["alignment_log_partition"]
