"""CUDA bridge for exact all-path FFBS sampling."""

from __future__ import annotations

import torch
from torch import Tensor

from ...sample_path import SampledPath
from ...validation import validate_sample_path_metadata
from ._config import DEFAULT_SAMPLE_PATH_CONFIG, SamplePathLaunchConfig
from ._extension import load_cuda_extension


@torch.no_grad()
def sample_path_cuda(
  transition_scores: Tensor,
  node_lengths: Tensor,
  uniforms: Tensor,
  *,
  launch_config: SamplePathLaunchConfig = DEFAULT_SAMPLE_PATH_CONFIG,
) -> SampledPath:
  """Run all-path alpha followed by compiled inverse-CDF backtrace."""
  validate_sample_path_metadata(transition_scores, node_lengths, uniforms)
  if not transition_scores.is_cuda:
    raise ValueError("all CUDA graph inputs must be CUDA tensors")
  nodes, lengths = load_cuda_extension().sample_path(
    transition_scores.contiguous(),
    node_lengths.contiguous(),
    uniforms.contiguous(),
    launch_config.forward,
    launch_config.sampling,
  )
  return SampledPath(nodes=nodes, lengths=lengths)


__all__ = ["SamplePathLaunchConfig", "sample_path_cuda"]
