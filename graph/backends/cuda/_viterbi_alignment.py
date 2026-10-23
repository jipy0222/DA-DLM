"""CUDA bridge for fixed-target Viterbi alignment."""

from __future__ import annotations

import torch
from torch import Tensor

from ...validation import validate_alignment_metadata
from ...viterbi_alignment import ViterbiAlignment
from ._config import DEFAULT_VITERBI_ALIGNMENT_CONFIG, ViterbiAlignmentLaunchConfig
from ._extension import load_cuda_extension
from ._transitions import dense_to_compact


@torch.no_grad()
def viterbi_alignment_cuda(
  emission_scores: Tensor,
  transition_scores: Tensor,
  node_lengths: Tensor,
  target_lengths: Tensor,
  *,
  launch_config: ViterbiAlignmentLaunchConfig = (
    DEFAULT_VITERBI_ALIGNMENT_CONFIG
  ),
) -> ViterbiAlignment:
  """Run the compiled fixed-target max-product and backtrace kernels."""
  validate_alignment_metadata(
    emission_scores,
    transition_scores,
    node_lengths,
    target_lengths,
  )
  if not emission_scores.is_cuda:
    raise ValueError("all CUDA graph inputs must be CUDA tensors")
  emissions = emission_scores.contiguous()
  compact_transitions = dense_to_compact(transition_scores).contiguous()
  scores, nodes = load_cuda_extension().viterbi_alignment(
    emissions,
    compact_transitions,
    node_lengths.contiguous(),
    target_lengths.contiguous(),
    launch_config.forward,
  )
  return ViterbiAlignment(scores=scores, nodes=nodes)


__all__ = [
  "ViterbiAlignmentLaunchConfig",
  "viterbi_alignment_cuda",
]
