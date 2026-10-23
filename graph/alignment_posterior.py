"""Pure PyTorch forward-backward alignment posterior reference.

The function uses dense transition scores with absolute destinations. Only
strictly forward edges participate in the dynamic programs; diagonal,
backward, and padded entries are ignored.

The public path handles expected ragged lengths and performs inexpensive
shape, dtype, device, and length checks.  Full value scans use the explicit
strict validators rather than running in every training-step invocation.
Valid score entries must be finite or ``-inf``; ``NaN`` and ``+inf`` are
outside the API contract and deliberately are not scanned for on the hot path.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

from ._forward import _fixed_length_alignment_forward_log_weights
from ._logspace import add_unless_impossible, stable_logsumexp
from .validation import mask_dense_dag, validate_alignment_metadata


def alignment_log_posterior(
  emission_scores: Tensor,
  transition_scores: Tensor,
  node_lengths: Tensor,
  target_lengths: Tensor,
) -> Tensor:
  """Return the fixed-length target-position/node alignment posterior.

  The returned ``[B, T_max, N_max]`` tensor contains
  ``log p(a_t = j | emissions, transitions, target_length)``.  A path starts
  at node 0, ends at ``node_lengths[b] - 1``, visits exactly
  ``target_lengths[b]`` nodes, and emits one target position at each visit.
  Every valid target-position slice is normalized over nodes.

  Padded and impossible alignments are ``-inf``.  If no path of the requested
  length reaches the terminal node, the entire sample is ``-inf`` with zero
  gradients.
  Callers express forced alignments by setting prohibited emission entries to
  ``-inf``; this function does not infer assignment constraints.
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
  valid_states = valid_targets.unsqueeze(2) & valid_nodes.unsqueeze(1)
  emissions = emission_scores.masked_fill(~valid_states, -torch.inf)

  all_alpha = _fixed_length_alignment_forward_log_weights(
    emissions,
    transitions,
  )

  end_nodes = node_lengths - 1
  end_times = target_lengths - 1
  terminal_beta = emissions.new_full((batch_size, max_node_count), -torch.inf)
  terminal_beta.scatter_(1, end_nodes.unsqueeze(1), 0)

  beta_next = emissions.new_full((batch_size, max_node_count), -torch.inf)
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


__all__ = ["alignment_log_posterior"]
