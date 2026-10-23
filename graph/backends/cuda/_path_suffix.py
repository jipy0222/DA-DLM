"""Inference-only CUDA bridge for all-path suffix log weights."""

from __future__ import annotations

import torch
from torch import Tensor

from ...validation import validate_transition_metadata
from ._config import DEFAULT_PATH_SUFFIX_CONFIG, PathSuffixLaunchConfig
from ._extension import load_cuda_extension


@torch.no_grad()
def path_suffix_log_partition_cuda(
  transition_scores: Tensor,
  node_lengths: Tensor,
  *,
  launch_config: PathSuffixLaunchConfig = DEFAULT_PATH_SUFFIX_CONFIG,
) -> Tensor:
  """Return every node's suffix log weight using FP64 internal accumulation.

  The result retains the score dtype. Each terminal has weight zero; padded
  nodes and nodes without a complete continuation have weight minus infinity.
  """
  validate_transition_metadata(transition_scores, node_lengths)
  if not transition_scores.is_cuda:
    raise ValueError("all CUDA graph inputs must be CUDA tensors")
  return load_cuda_extension().path_suffix_partition(
    transition_scores.contiguous(), node_lengths.contiguous(),
    launch_config.forward,
  )


__all__ = ["PathSuffixLaunchConfig", "path_suffix_log_partition_cuda"]
