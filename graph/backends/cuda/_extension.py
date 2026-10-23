"""Lazy JIT loader for the graph CUDA extension.

Compilation is triggered only by an explicit CUDA graph call or warm-up.  The
visible CUDA devices determine architecture flags through PyTorch's
``cpp_extension`` defaults.
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import cast

import torch
from torch.utils.cpp_extension import load

from ._availability import inspect_cuda_environment
from ._extension_types import CudaExtension

_EXTENSION: CudaExtension | None = None
_EXTENSION_NAME: str | None = None
_EXTENSION_LOCK = Lock()


def visible_cuda_architectures() -> tuple[str, ...]:
  """Return stable architecture labels for all visible CUDA devices."""
  if not torch.cuda.is_available():
    return ()
  capabilities = {
    torch.cuda.get_device_capability(index)
    for index in range(torch.cuda.device_count())
  }
  return tuple(
    f"{major}.{minor}" for major, minor in sorted(capabilities)
  )


def cuda_extension_name() -> str:
  """Build the internal cache/module name from automatically detected GPUs."""
  architectures = visible_cuda_architectures()
  if not architectures:
    raise RuntimeError(
      "CUDA graph backend is unavailable: no visible CUDA architecture"
    )
  architecture_tag = "_".join(
    f"sm{architecture.replace('.', '')}" for architecture in architectures
  )
  return f"dadlm_graph_cuda_{architecture_tag}"


def cuda_extension_is_loaded() -> bool:
  """Return whether this process has already loaded the graph extension."""
  return _EXTENSION is not None


def loaded_cuda_extension_name() -> str | None:
  """Return the loaded module name without triggering compilation."""
  return _EXTENSION_NAME


def load_cuda_extension() -> CudaExtension:
  """Build or load the extension lazily on explicit CUDA use.

  PyTorch supplies an inter-process build lock.  The local lock additionally
  prevents two threads in one Python process from entering the loader at once.
  The loaded library is reused for this process; restart Python after editing
  native sources so the build system checks their dependencies again.
  Restart Python after a failed build before retrying the loader as well.
  """
  global _EXTENSION, _EXTENSION_NAME
  if _EXTENSION is not None:
    return _EXTENSION

  with _EXTENSION_LOCK:
    if _EXTENSION is not None:
      return _EXTENSION

    environment = inspect_cuda_environment()
    reason = environment.unavailable_reason()
    if reason is not None:
      raise RuntimeError(f"CUDA graph backend is unavailable: {reason}")

    architectures = visible_cuda_architectures()
    extension_name = cuda_extension_name()
    source_directory = Path(__file__).resolve().parent / "csrc"
    try:
      extension = load(
        # Keep automatically detected GPU architectures in separate build caches.
        name=extension_name,
        sources=[
          str(source_directory / "graph_ops.cpp"),
          str(source_directory / "alignment_partition.cu"),
          str(source_directory / "alignment_posterior.cu"),
          str(source_directory / "path_partition.cu"),
          str(source_directory / "viterbi_alignment.cu"),
          str(source_directory / "viterbi_path.cu"),
          str(source_directory / "sample_path.cu"),
          str(source_directory / "sample_alignment.cu"),
          str(source_directory / "logsoftmax_gather.cu"),
        ],
        extra_cflags=["-O3"],
        # PyTorch derives architecture flags from every visible device when
        # TORCH_CUDA_ARCH_LIST is not explicitly set.
        extra_cuda_cflags=["-O3"],
        extra_include_paths=[str(source_directory)],
        with_cuda=True,
        verbose=False,
      )
    except Exception as error:
      architecture_text = ", ".join(architectures)
      raise RuntimeError(
        "failed to JIT-build the CUDA graph extension for automatically "
        f"detected architecture(s) {architecture_text}; toolkit root is "
        f"{environment.toolkit_root!r}"
      ) from error
    _EXTENSION = cast(CudaExtension, extension)
    _EXTENSION_NAME = extension_name
    return cast(CudaExtension, extension)


__all__ = [
  "cuda_extension_is_loaded",
  "cuda_extension_name",
  "load_cuda_extension",
  "loaded_cuda_extension_name",
  "visible_cuda_architectures",
]
