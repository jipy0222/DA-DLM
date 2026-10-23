"""Environment checks for the optional CUDA graph backend.

These checks report CUDA runtime and source-build toolchain availability
without compiling or loading the extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.cpp_extension import CUDA_HOME


@dataclass(frozen=True)
class CudaEnvironment:
  """Observed CUDA runtime/toolchain state without triggering a build."""

  runtime_available: bool
  toolkit_root: str | None
  nvcc_available: bool

  @property
  def ready_for_source_build(self) -> bool:
    return (
      self.runtime_available
      and self.toolkit_root is not None
      and self.nvcc_available
    )

  def unavailable_reason(self) -> str | None:
    if not self.runtime_available:
      return "torch.cuda.is_available() is false"
    if self.toolkit_root is None:
      return "torch.utils.cpp_extension.CUDA_HOME is unset"
    if not self.nvcc_available:
      return f"nvcc was not found under {self.toolkit_root!r}"
    return None


def inspect_cuda_environment() -> CudaEnvironment:
  """Return CUDA availability without compiling or loading graph kernels."""
  toolkit_root = None if CUDA_HOME is None else str(CUDA_HOME)
  nvcc_available = bool(
    toolkit_root is not None
    and (Path(toolkit_root) / "bin" / "nvcc").is_file()
  )
  return CudaEnvironment(
    runtime_available=torch.cuda.is_available(),
    toolkit_root=toolkit_root,
    nvcc_available=nvcc_available,
  )


__all__ = ["CudaEnvironment", "inspect_cuda_environment"]
