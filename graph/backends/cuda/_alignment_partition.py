"""Autograd bridge for the fixed-target alignment partition CUDA kernel."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor
from torch.autograd import Function

from ...validation import validate_alignment_metadata
from ._autograd_types import AlignmentPartitionContext
from ._config import DEFAULT_ALIGNMENT_PARTITION_CONFIG, AlignmentPartitionLaunchConfig
from ._extension import load_cuda_extension
from ._transitions import dense_to_compact


class _AlignmentLogPartitionCuda(Function):
  @staticmethod
  def forward(
    ctx: AlignmentPartitionContext,
    emission_scores: Tensor,
    compact_transition_scores: Tensor,
    node_lengths: Tensor,
    target_lengths: Tensor,
    require_gradient: bool,
    forward_config: int,
    emission_backward_config: int,
    transition_backward_config: int,
  ) -> Tensor:
    extension = load_cuda_extension()
    emissions = emission_scores.contiguous()
    transitions = compact_transition_scores.contiguous()
    lengths = node_lengths.contiguous()
    targets = target_lengths.contiguous()
    alpha, beta = extension.alignment_partition_forward(
      emissions,
      transitions,
      lengths,
      targets,
      require_gradient,
      forward_config,
    )
    if require_gradient:
      # Use the same partition representation as the analytical backward:
      # beta at the fixed start state equals alpha at the ragged terminal state.
      result = beta[:, 0, 0].clone()
      ctx.save_for_backward(
        alpha, beta, emissions, transitions, lengths, targets)
      ctx.emission_backward_config = emission_backward_config
      ctx.transition_backward_config = transition_backward_config
    else:
      batch_index = torch.arange(
        emissions.shape[0], device=emissions.device)
      result = alpha[batch_index, targets - 1, lengths - 1]
    return result

  # PyTorch's generic backward stub cannot express this input-specific gradient tuple.
  @staticmethod
  def backward(ctx: AlignmentPartitionContext, grad_output: Tensor) -> tuple[Tensor, Tensor, None, None, None, None, None, None]:  # ty: ignore[invalid-method-override]
    alpha, beta, emissions, transitions, lengths, targets = ctx.saved_tensors
    extension = load_cuda_extension()
    grad_emissions, grad_transitions = (
      extension.alignment_partition_backward(
        grad_output.contiguous(),
        alpha,
        beta,
        emissions,
        transitions,
        lengths,
        targets,
        ctx.emission_backward_config,
        ctx.transition_backward_config,
      )
    )
    return grad_emissions, grad_transitions, None, None, None, None, None, None


def alignment_log_partition_cuda(
  emission_scores: Tensor,
  transition_scores: Tensor,
  node_lengths: Tensor,
  target_lengths: Tensor,
  *,
  launch_config: AlignmentPartitionLaunchConfig = (
    DEFAULT_ALIGNMENT_PARTITION_CONFIG
  ),
) -> Tensor:
  """Run compiled fixed-target sum-product forward and analytical backward."""
  validate_alignment_metadata(
    emission_scores,
    transition_scores,
    node_lengths,
    target_lengths,
  )
  if not emission_scores.is_cuda:
    raise ValueError("all CUDA alignment partition inputs must be CUDA tensors")
  require_gradient = torch.is_grad_enabled() and (
    emission_scores.requires_grad or transition_scores.requires_grad
  )
  compact_transitions = dense_to_compact(transition_scores)
  return cast('Tensor', _AlignmentLogPartitionCuda.apply(
    emission_scores,
    compact_transitions,
    node_lengths,
    target_lengths,
    require_gradient,
    launch_config.forward,
    launch_config.emission_backward,
    launch_config.transition_backward,
  ))


__all__ = [
  "AlignmentPartitionLaunchConfig",
  "alignment_log_partition_cuda",
]
