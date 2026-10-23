"""The pybind module interface exposed by csrc/graph_ops.cpp."""

from __future__ import annotations

from typing import Protocol

from torch import Tensor


class CudaExtension(Protocol):
    __name__: str
    __file__: str

    def alignment_partition_forward(self, emissions: Tensor, transitions: Tensor,
        lengths: Tensor, targets: Tensor, require_gradient: bool,
        config: int) -> tuple[Tensor, Tensor]: ...
    def alignment_partition_backward(self, gradient: Tensor, alpha: Tensor,
        beta: Tensor, emissions: Tensor, transitions: Tensor, lengths: Tensor,
        targets: Tensor, emission_config: int,
        transition_config: int) -> tuple[Tensor, Tensor]: ...
    def alignment_posterior_forward(self, emissions: Tensor, transitions: Tensor,
        lengths: Tensor, targets: Tensor, config: int) -> tuple[Tensor, Tensor, Tensor]: ...
    def alignment_posterior_backward(self, gradient: Tensor, posterior: Tensor,
        alpha: Tensor, beta: Tensor, emissions: Tensor, transitions: Tensor,
        lengths: Tensor, targets: Tensor, config: int) -> tuple[Tensor, Tensor]: ...
    def path_partition_forward(self, transitions: Tensor, lengths: Tensor,
        config: int) -> tuple[Tensor, Tensor]: ...
    def path_partition_backward(self, gradient: Tensor, alpha: Tensor,
        transitions: Tensor, lengths: Tensor, config: int) -> Tensor: ...
    def path_suffix_partition(self, transitions: Tensor, lengths: Tensor,
        config: int) -> Tensor: ...
    def viterbi_path(self, scores: Tensor, transitions: Tensor, lengths: Tensor,
        config: int) -> tuple[Tensor, Tensor, Tensor]: ...
    def viterbi_alignment(self, emissions: Tensor, transitions: Tensor,
        lengths: Tensor, targets: Tensor, config: int) -> tuple[Tensor, Tensor]: ...
    def sample_path(self, transitions: Tensor, lengths: Tensor, uniforms: Tensor,
        forward_config: int, sampling_config: int) -> tuple[Tensor, Tensor]: ...
    def sample_alignment(self, emissions: Tensor, transitions: Tensor,
        lengths: Tensor, targets: Tensor, uniforms: Tensor, forward_config: int,
        sampling_config: int) -> Tensor: ...
    def logsoftmax_gather(self, workspace: Tensor, indices: Tensor,
        require_gradient: bool, threads: int) -> Tensor: ...
