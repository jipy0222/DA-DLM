"""Static contracts for contexts created by PyTorch's autograd engine."""

from __future__ import annotations

from typing import Protocol

from torch import Tensor, dtype


class SavedTensors(Protocol):
    saved_tensors: tuple[Tensor, ...]
    def save_for_backward(self, *tensors: Tensor) -> None: ...


class AlignmentPartitionContext(SavedTensors, Protocol):
    emission_backward_config: int
    transition_backward_config: int


class PartitionContext(SavedTensors, Protocol):
    backward_config: int


class GatherContext(SavedTensors, Protocol):
    input_dtype: dtype
    backward_consumed: bool
