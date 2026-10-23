"""CUDA backends for graph operations.

Differentiable partitions and posteriors support first-order autograd only.
Suffix weights and path selection are inference-only.

Importing this module is side-effect free: no extension is compiled or loaded.
The shared extension is JIT-built for automatically detected visible GPU
architectures on first use. Large jobs may call :func:`warmup_cuda` before the
first batch to trigger exactly the same loader proactively.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._alignment_partition import (
    alignment_log_partition_cuda as alignment_log_partition,
)
from ._alignment_posterior import (
    alignment_log_posterior_cuda as alignment_log_posterior,
)
from ._availability import CudaEnvironment, inspect_cuda_environment
from ._path_partition import path_log_partition_cuda as path_log_partition
from ._path_suffix import path_suffix_log_partition_cuda as path_suffix_log_partition
from ._sample_alignment import sample_alignment_cuda as sample_alignment
from ._sample_path import sample_path_cuda as sample_path
from ._viterbi_alignment import viterbi_alignment_cuda as viterbi_alignment
from ._viterbi_path import viterbi_path_cuda as viterbi_path

if TYPE_CHECKING:
    from .warmup import CudaWarmupResult


def warmup_cuda() -> CudaWarmupResult:
  """Lazily import and run the optional pre-first-batch JIT warm-up."""
  from .warmup import warmup_cuda as _warmup_cuda

  return _warmup_cuda()


__all__ = [
  "CudaEnvironment",
  "alignment_log_partition",
  "alignment_log_posterior",
  "inspect_cuda_environment",
  "path_log_partition",
  "path_suffix_log_partition",
  "sample_alignment",
  "sample_path",
  "viterbi_alignment",
  "viterbi_path",
  "warmup_cuda",
]
