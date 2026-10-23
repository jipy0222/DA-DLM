"""Training-start hook for the graph backend.

Call :func:`prepare_graph_runtime` after CUDA devices are assigned and before
the first training batch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from graph.runtime import GraphBackend, validate_backend

if TYPE_CHECKING:
    from graph.backends.cuda.warmup import CudaWarmupResult


def prepare_graph_runtime(
    *,
    backend: GraphBackend = "auto",
    warmup_before_training: bool = True,
) -> CudaWarmupResult | None:
    """Optionally JIT-build/load graph CUDA kernels before the first batch."""
    selected = validate_backend(backend)
    if not isinstance(warmup_before_training, bool):
        raise TypeError("warmup_before_training must be a bool")
    if not warmup_before_training or selected == "pytorch":
        return None

    import torch

    if selected == "auto" and not torch.cuda.is_available():
        return None

    from graph.backends.cuda import warmup_cuda

    return warmup_cuda()


__all__ = ["prepare_graph_runtime"]
