"""CUDA bridge for exact fixed-target alignment FFBS sampling."""

from __future__ import annotations

import torch
from torch import Tensor

from ...sample_alignment import SampledAlignment
from ...validation import validate_sample_alignment_metadata
from ._config import DEFAULT_SAMPLE_ALIGNMENT_CONFIG, SampleAlignmentLaunchConfig
from ._extension import load_cuda_extension
from ._transitions import dense_to_compact


@torch.no_grad()
def sample_alignment_cuda(
  emission_scores: Tensor,
  transition_scores: Tensor,
  node_lengths: Tensor,
  target_lengths: Tensor,
  uniforms: Tensor,
  *,
  launch_config: SampleAlignmentLaunchConfig = (
    DEFAULT_SAMPLE_ALIGNMENT_CONFIG
  ),
) -> SampledAlignment:
  """Run fixed-target alpha followed by compiled inverse-CDF FFBS."""
  validate_sample_alignment_metadata(
    emission_scores,
    transition_scores,
    node_lengths,
    target_lengths,
    uniforms,
  )
  if not emission_scores.is_cuda:
    raise ValueError("all CUDA graph inputs must be CUDA tensors")
  compact_transitions = dense_to_compact(transition_scores).contiguous()
  nodes = load_cuda_extension().sample_alignment(
    emission_scores.contiguous(),
    compact_transitions,
    node_lengths.contiguous(),
    target_lengths.contiguous(),
    uniforms.contiguous(),
    launch_config.forward,
    launch_config.sampling,
  )
  return SampledAlignment(nodes=nodes)


__all__ = ["SampleAlignmentLaunchConfig", "sample_alignment_cuda"]
