"""Autograd bridge for the all-path partition CUDA kernels."""

from __future__ import annotations

from typing import cast

from torch import Tensor
from torch.autograd import Function

from ...validation import validate_transition_metadata
from ._autograd_types import PartitionContext
from ._config import DEFAULT_PATH_PARTITION_CONFIG, PathPartitionLaunchConfig
from ._extension import load_cuda_extension


class _PathLogPartitionCuda(Function):
  @staticmethod
  def forward(
    ctx: PartitionContext,
    transition_scores: Tensor,
    node_lengths: Tensor,
    forward_config: int,
    backward_config: int,
  ) -> Tensor:
    transitions = transition_scores.contiguous()
    lengths = node_lengths.contiguous()
    partition, alpha = load_cuda_extension().path_partition_forward(
      transitions,
      lengths,
      forward_config,
    )
    if transition_scores.requires_grad:
      ctx.save_for_backward(alpha, transitions, lengths)
      ctx.backward_config = backward_config
    return partition

  # PyTorch's generic backward stub cannot express this input-specific gradient tuple.
  @staticmethod
  def backward(ctx: PartitionContext, grad_output: Tensor) -> tuple[Tensor, None, None, None]:  # ty: ignore[invalid-method-override]
    alpha, transitions, lengths = ctx.saved_tensors
    transition_gradient = load_cuda_extension().path_partition_backward(
      grad_output.contiguous(),
      alpha,
      transitions,
      lengths,
      ctx.backward_config,
    )
    return transition_gradient, None, None, None


def path_log_partition_cuda(
  transition_scores: Tensor,
  node_lengths: Tensor,
  *,
  launch_config: PathPartitionLaunchConfig = DEFAULT_PATH_PARTITION_CONFIG,
) -> Tensor:
  """Run compiled all-path sum-product forward and analytical backward."""
  validate_transition_metadata(transition_scores, node_lengths)
  if not transition_scores.is_cuda:
    raise ValueError("all CUDA graph inputs must be CUDA tensors")
  return cast('Tensor', _PathLogPartitionCuda.apply(
    transition_scores,
    node_lengths,
    launch_config.forward,
    launch_config.backward,
  ))


__all__ = ["PathPartitionLaunchConfig", "path_log_partition_cuda"]
