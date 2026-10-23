"""Pure PyTorch reference for the all-paths transition partition."""

from __future__ import annotations

import torch
from torch import Tensor

from ._forward import _all_paths_forward_log_weights
from .validation import mask_dense_dag, validate_transition_metadata


def path_log_partition(
  transition_scores: Tensor,
  node_lengths: Tensor,
) -> Tensor:
  """Sum transition weights over all complete DAG paths in log space.

  Args:
    transition_scores: Dense ``[B, N_max, N_max]`` log scores or potentials.
      Entry ``[b, i, j]`` represents edge ``i -> j``. Only strict-forward
      edges within each sample's node length participate.
    node_lengths: ``[B]`` long tensor. A complete path starts at node 0 and
      ends at ``node_lengths[b] - 1``.

  Returns:
    A ``[B]`` tensor containing
    ``log sum_path exp(sum_edge transition_score)`` over every complete path,
    without constraining path length.  No emissions, loss sign, or reduction
    are applied.  An unreachable terminal returns ``-inf`` with zero gradient.

  Locally normalized transition log probabilities produce a partition of zero
  when every valid source distribution is normalized.  Unnormalized CRF
  potentials generally produce a nonzero partition.  The normal call performs
  lightweight metadata/length validation; full value scans are available via
  ``strict_validate_transition_inputs``.
  """
  batch_size, _ = validate_transition_metadata(
    transition_scores,
    node_lengths,
  )
  transitions, _ = mask_dense_dag(transition_scores, node_lengths)
  alpha = _all_paths_forward_log_weights(transitions, node_lengths)
  batch_index = torch.arange(batch_size, device=transition_scores.device)
  return alpha[batch_index, node_lengths - 1]


@torch.no_grad()
def path_suffix_log_partition(
  transition_scores: Tensor,
  node_lengths: Tensor,
) -> Tensor:
  """Return the all-paths log weight from every node to its row's terminal.

  The result has shape ``[B, N_max]``. Each terminal
  ``node_lengths[b] - 1`` has weight zero; padded nodes and nodes without a
  complete forward continuation have weight ``-inf``. This inference-only
  PyTorch reference runs on the input device, including CUDA, without a
  custom kernel or a fixed path-length constraint.
  """
  _, max_node_count = validate_transition_metadata(
    transition_scores,
    node_lengths,
  )
  transitions, _ = mask_dense_dag(transition_scores, node_lengths)
  beta = transitions.new_full(
    (transition_scores.shape[0], max_node_count), -torch.inf)
  beta.scatter_(1, (node_lengths - 1).unsqueeze(1), 0)

  for source in range(max_node_count - 2, -1, -1):
    continuation = torch.logsumexp(
      transitions[:, source, source + 1:] + beta[:, source + 1:],
      dim=1,
    )
    beta[:, source] = torch.where(
      node_lengths == source + 1,
      torch.zeros_like(continuation),
      continuation,
    )
  return beta


__all__ = ["path_log_partition", "path_suffix_log_partition"]
