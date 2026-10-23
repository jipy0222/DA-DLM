"""DADLM graph semantics and pure PyTorch reference algorithms.

This package remains importable without compiling optional CUDA extensions.
"""

from .alignment_partition import alignment_log_partition
from .alignment_posterior import alignment_log_posterior
from .path_partition import path_log_partition, path_suffix_log_partition
from .sample_alignment import SampledAlignment, sample_alignment
from .sample_path import SampledPath, sample_path
from .validation import (
  strict_validate_alignment_inputs,
  strict_validate_sample_alignment_inputs,
  strict_validate_sample_path_inputs,
  strict_validate_scored_path_inputs,
  strict_validate_transition_inputs,
)
from .viterbi_alignment import ViterbiAlignment, viterbi_alignment
from .viterbi_path import ViterbiPath, viterbi_path

__all__ = [
  "SampledAlignment",
  "SampledPath",
  "ViterbiAlignment",
  "ViterbiPath",
  "alignment_log_partition",
  "alignment_log_posterior",
  "path_log_partition",
  "path_suffix_log_partition",
  "sample_alignment",
  "sample_path",
  "strict_validate_alignment_inputs",
  "strict_validate_sample_alignment_inputs",
  "strict_validate_sample_path_inputs",
  "strict_validate_scored_path_inputs",
  "strict_validate_transition_inputs",
  "viterbi_alignment",
  "viterbi_path",
]
