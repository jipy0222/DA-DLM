"""Autograd bridge for fixed-target alignment log-posteriors on CUDA."""

from __future__ import annotations

from typing import cast

from torch import Tensor
from torch.autograd import Function

from ...validation import validate_alignment_metadata
from ._autograd_types import PartitionContext
from ._config import DEFAULT_ALIGNMENT_POSTERIOR_CONFIG, AlignmentPosteriorLaunchConfig
from ._extension import load_cuda_extension
from ._transitions import dense_to_compact


class _AlignmentLogPosteriorCuda(Function):
  @staticmethod
  def forward(
    ctx: PartitionContext,
    emission_scores: Tensor,
    compact_transition_scores: Tensor,
    node_lengths: Tensor,
    target_lengths: Tensor,
    forward_config: int,
    backward_config: int,
  ) -> Tensor:
    extension = load_cuda_extension()
    emissions = emission_scores.contiguous()
    transitions = compact_transition_scores.contiguous()
    lengths = node_lengths.contiguous()
    targets = target_lengths.contiguous()
    posterior, alpha, beta = extension.alignment_posterior_forward(
      emissions,
      transitions,
      lengths,
      targets,
      forward_config,
    )
    ctx.save_for_backward(
      posterior, alpha, beta, emissions, transitions, lengths, targets)
    ctx.backward_config = backward_config
    return posterior

  # PyTorch's generic backward stub cannot express this input-specific gradient tuple.
  @staticmethod
  def backward(ctx: PartitionContext, grad_output: Tensor) -> tuple[Tensor, Tensor, None, None, None, None]:  # ty: ignore[invalid-method-override]
    posterior, alpha, beta, emissions, transitions, lengths, targets = (
      ctx.saved_tensors
    )
    grad_emissions, grad_transitions = (
      load_cuda_extension().alignment_posterior_backward(
        grad_output.contiguous(),
        posterior,
        alpha,
        beta,
        emissions,
        transitions,
        lengths,
        targets,
        ctx.backward_config,
      )
    )
    return grad_emissions, grad_transitions, None, None, None, None


def alignment_log_posterior_cuda(
  emission_scores: Tensor,
  transition_scores: Tensor,
  node_lengths: Tensor,
  target_lengths: Tensor,
  *,
  launch_config: AlignmentPosteriorLaunchConfig = (
    DEFAULT_ALIGNMENT_POSTERIOR_CONFIG
  ),
) -> Tensor:
  """Run compiled forward-backward with an arbitrary log-posterior VJP."""
  validate_alignment_metadata(
    emission_scores,
    transition_scores,
    node_lengths,
    target_lengths,
  )
  if not emission_scores.is_cuda:
    raise ValueError("all CUDA graph inputs must be CUDA tensors")
  compact_transitions = dense_to_compact(transition_scores)
  return cast('Tensor', _AlignmentLogPosteriorCuda.apply(
    emission_scores,
    compact_transitions,
    node_lengths,
    target_lengths,
    launch_config.forward,
    launch_config.backward,
  ))


__all__ = ["AlignmentPosteriorLaunchConfig", "alignment_log_posterior_cuda"]
