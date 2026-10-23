"""Internal forward dynamic programs shared by DAG references."""

from __future__ import annotations

import torch
from torch import Tensor

from ._logspace import add_unless_impossible, stable_logsumexp


def _fixed_length_alignment_forward_log_weights(
  emission_scores: Tensor,
  transition_scores: Tensor,
) -> Tensor:
  """Return fixed-target alpha values with shape ``[B, T, N]``.

  Inputs must already have padded states and non-forward transitions masked to
  ``-inf``. Public callers own validation and masking.
  """
  batch_size, max_target_length, max_node_count = emission_scores.shape
  alpha = emission_scores.new_full(
    (batch_size, max_node_count), -torch.inf)
  alpha[:, 0] = emission_scores[:, 0, 0]
  alpha_steps = [alpha]

  for target_position in range(1, max_target_length):
    incoming = stable_logsumexp(
      alpha.unsqueeze(2) + transition_scores,
      dim=1,
    ).squeeze(1)
    alpha = add_unless_impossible(
      incoming,
      emission_scores[:, target_position, :],
    )
    alpha_steps.append(alpha)
  return torch.stack(alpha_steps, dim=1)


def _all_paths_forward_log_weights(
  transition_scores: Tensor,
  node_lengths: Tensor,
) -> Tensor:
  """Return all-length alpha values with shape ``[B, N]``.

  ``transition_scores`` must already have padded and non-forward entries
  masked to ``-inf``. Public callers own validation and masking.
  """
  batch_size, max_node_count = transition_scores.shape[:2]
  # Assemble the dense edge gradient once from the column views.
  transition_columns = transition_scores.unbind(dim=2)
  alpha_columns = [transition_scores.new_zeros(batch_size)]
  for destination in range(1, max_node_count):
    previous_alpha = torch.stack(alpha_columns, dim=1)
    incoming = stable_logsumexp(
      previous_alpha + transition_columns[destination][:, :destination],
      dim=1,
    ).squeeze(1)
    destination_valid = destination < node_lengths
    alpha_destination = torch.where(
      destination_valid & ~torch.isneginf(incoming),
      incoming,
      torch.full_like(incoming, -torch.inf),
    )
    alpha_columns.append(alpha_destination)
  return torch.stack(alpha_columns, dim=1)
