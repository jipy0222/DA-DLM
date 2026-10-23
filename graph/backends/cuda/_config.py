"""Validated internal launch configurations for CUDA graph kernels."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlignmentPartitionLaunchConfig:
  """Forward/backward kernel variants for fixed-target alignment partition."""

  forward: int = 1
  emission_backward: int = 2
  transition_backward: int = 2

  def __post_init__(self) -> None:
    if self.forward not in {1, 2, 3, 4}:
      raise ValueError("alignment-partition forward config must be in 1..4")
    if self.emission_backward not in {1, 2}:
      raise ValueError(
        "alignment-partition emission-backward config must be in 1..2"
      )
    if self.transition_backward not in {1, 2, 3}:
      raise ValueError(
        "alignment-partition transition-backward config must be in 1..3"
      )


DEFAULT_ALIGNMENT_PARTITION_CONFIG = AlignmentPartitionLaunchConfig()


@dataclass(frozen=True)
class AlignmentPosteriorLaunchConfig:
  """Fixed-target alpha/beta and arbitrary-posterior-VJP kernel variants."""

  forward: int = 3
  backward: int = 4

  def __post_init__(self) -> None:
    if self.forward not in {1, 2, 3, 4}:
      raise ValueError("alignment-posterior forward config must be in 1..4")
    if self.backward not in {1, 2, 3, 4}:
      raise ValueError("alignment-posterior backward config must be in 1..4")


DEFAULT_ALIGNMENT_POSTERIOR_CONFIG = AlignmentPosteriorLaunchConfig()


@dataclass(frozen=True)
class PathPartitionLaunchConfig:
  """Persistent-warp variants for all-path sum-product reduction."""

  forward: int = 4
  backward: int = 4

  def __post_init__(self) -> None:
    if self.forward not in {1, 2, 3, 4}:
      raise ValueError("path-partition forward config must be in 1..4")
    if self.backward not in {1, 2, 3, 4}:
      raise ValueError("path-partition backward config must be in 1..4")


DEFAULT_PATH_PARTITION_CONFIG = PathPartitionLaunchConfig()


@dataclass(frozen=True)
class PathSuffixLaunchConfig:
  """Persistent-warp variants for the all-path suffix reduction."""

  forward: int = 4

  def __post_init__(self) -> None:
    if self.forward not in {1, 2, 3, 4}:
      raise ValueError("path-suffix forward config must be in 1..4")


DEFAULT_PATH_SUFFIX_CONFIG = PathSuffixLaunchConfig()


@dataclass(frozen=True)
class ViterbiAlignmentLaunchConfig:
  """Kernel variant for fixed-target Viterbi alignment."""

  forward: int = 3

  def __post_init__(self) -> None:
    if self.forward not in {1, 2, 3, 4}:
      raise ValueError("Viterbi-alignment forward config must be in 1..4")


DEFAULT_VITERBI_ALIGNMENT_CONFIG = ViterbiAlignmentLaunchConfig()


@dataclass(frozen=True)
class ViterbiPathLaunchConfig:
  """Persistent-block variants for all-path max-product reduction."""

  forward: int = 4

  def __post_init__(self) -> None:
    if self.forward not in {1, 2, 3, 4}:
      raise ValueError("Viterbi-path forward config must be in 1..4")


DEFAULT_VITERBI_PATH_CONFIG = ViterbiPathLaunchConfig()


@dataclass(frozen=True)
class SamplePathLaunchConfig:
  """Independent all-path alpha and FFBS predecessor-reduction variants."""

  forward: int = 4
  sampling: int = 4

  def __post_init__(self) -> None:
    if self.forward not in {1, 2, 3, 4}:
      raise ValueError("sample-path forward config must be in 1..4")
    if self.sampling not in {1, 2, 3, 4}:
      raise ValueError("sample-path sampling config must be in 1..4")


DEFAULT_SAMPLE_PATH_CONFIG = SamplePathLaunchConfig()


@dataclass(frozen=True)
class SampleAlignmentLaunchConfig:
  """Independent fixed-target alpha and FFBS kernel variants."""

  forward: int = 4
  sampling: int = 4

  def __post_init__(self) -> None:
    if self.forward not in {1, 2, 3, 4}:
      raise ValueError("sample-alignment forward config must be in 1..4")
    if self.sampling not in {1, 2, 3, 4}:
      raise ValueError("sample-alignment sampling config must be in 1..4")


DEFAULT_SAMPLE_ALIGNMENT_CONFIG = SampleAlignmentLaunchConfig()


@dataclass(frozen=True)
class LogSoftmaxGatherLaunchConfig:
  """Block size for rowwise log-softmax and selected-token gather."""

  threads: int = 1024

  def __post_init__(self) -> None:
    if self.threads not in {128, 256, 512, 1024}:
      raise ValueError(
        "log-softmax-gather threads must be 128, 256, 512, or 1024"
      )


DEFAULT_LOG_SOFTMAX_GATHER_CONFIG = LogSoftmaxGatherLaunchConfig()


__all__ = [
  "AlignmentPartitionLaunchConfig",
  "AlignmentPosteriorLaunchConfig",
  "DEFAULT_ALIGNMENT_PARTITION_CONFIG",
  "DEFAULT_ALIGNMENT_POSTERIOR_CONFIG",
  "DEFAULT_LOG_SOFTMAX_GATHER_CONFIG",
  "DEFAULT_PATH_PARTITION_CONFIG",
  "DEFAULT_PATH_SUFFIX_CONFIG",
  "DEFAULT_SAMPLE_ALIGNMENT_CONFIG",
  "DEFAULT_SAMPLE_PATH_CONFIG",
  "DEFAULT_VITERBI_ALIGNMENT_CONFIG",
  "DEFAULT_VITERBI_PATH_CONFIG",
  "LogSoftmaxGatherLaunchConfig",
  "PathPartitionLaunchConfig",
  "PathSuffixLaunchConfig",
  "SampleAlignmentLaunchConfig",
  "SamplePathLaunchConfig",
  "ViterbiAlignmentLaunchConfig",
  "ViterbiPathLaunchConfig",
]
