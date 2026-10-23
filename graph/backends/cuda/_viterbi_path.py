"""CUDA bridge for maximum-score complete DAG paths."""

from __future__ import annotations

import torch
from torch import Tensor

from ...validation import validate_scored_path_metadata
from ...viterbi_path import ViterbiPath
from ._config import DEFAULT_VITERBI_PATH_CONFIG, ViterbiPathLaunchConfig
from ._extension import load_cuda_extension


@torch.no_grad()
def viterbi_path_cuda(
  node_scores: Tensor,
  transition_scores: Tensor,
  node_lengths: Tensor,
  *,
  launch_config: ViterbiPathLaunchConfig = DEFAULT_VITERBI_PATH_CONFIG,
) -> ViterbiPath:
  """Run the compiled all-path max-product and backtrace kernels."""
  validate_scored_path_metadata(
    node_scores,
    transition_scores,
    node_lengths,
  )
  if not node_scores.is_cuda:
    raise ValueError("all CUDA graph inputs must be CUDA tensors")
  scores, nodes, lengths = load_cuda_extension().viterbi_path(
    node_scores.contiguous(),
    transition_scores.contiguous(),
    node_lengths.contiguous(),
    launch_config.forward,
  )
  return ViterbiPath(scores=scores, nodes=nodes, lengths=lengths)


__all__ = [
  "ViterbiPathLaunchConfig",
  "viterbi_path_cuda",
]
