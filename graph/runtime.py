"""Training/inference adapter for PyTorch and custom-CUDA graph backends.

The mathematical functions directly under :mod:`graph` remain independent
PyTorch references. Model and objective code should call this module so the
existing CUDA-backed operations use compiled kernels by default and
low-precision model outputs enter an explicit FP32 graph-DP precision island.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import ContextManager, Literal, cast

import torch
from torch import Tensor

from .alignment_partition import (
  alignment_log_partition as _pytorch_alignment_log_partition,
)
from .alignment_posterior import (
  alignment_log_posterior as _pytorch_alignment_log_posterior,
)
from .path_partition import path_log_partition as _pytorch_path_log_partition
from .path_partition import (
  path_suffix_log_partition as _pytorch_path_suffix_log_partition,
)
from .sample_alignment import SampledAlignment
from .sample_alignment import sample_alignment as _pytorch_sample_alignment
from .sample_path import SampledPath
from .sample_path import sample_path as _pytorch_sample_path
from .viterbi_alignment import ViterbiAlignment
from .viterbi_alignment import viterbi_alignment as _pytorch_viterbi_alignment
from .viterbi_path import ViterbiPath
from .viterbi_path import viterbi_path as _pytorch_viterbi_path

GraphBackend = Literal["auto", "pytorch", "cuda"]
_VALID_BACKENDS = frozenset({"auto", "pytorch", "cuda"})
_LOW_PRECISION_DTYPES = frozenset({torch.float16, torch.bfloat16})
_GRAPH_DTYPES = frozenset({
  torch.float16,
  torch.bfloat16,
  torch.float32,
  torch.float64,
})


def validate_backend(backend: str) -> GraphBackend:
  """Validate and narrow a public graph backend name."""
  if backend not in _VALID_BACKENDS:
    choices = ", ".join(sorted(_VALID_BACKENDS))
    raise ValueError(
      f"graph backend must be one of {choices}, got {backend!r}"
    )
  return cast(GraphBackend, backend)


def _resolve_backend(backend: str, primary_score: Tensor) -> GraphBackend:
  selected = validate_backend(backend)
  if selected == "auto":
    return "cuda" if primary_score.is_cuda else "pytorch"
  if selected == "cuda" and not primary_score.is_cuda:
    raise ValueError("backend='cuda' requires CUDA score tensors")
  return selected


def _graph_compute_dtype(primary_score: Tensor, *scores: Tensor) -> torch.dtype:
  score_dtype = primary_score.dtype
  if score_dtype not in _GRAPH_DTYPES:
    raise TypeError(
      "graph runtime scores support float16, bfloat16, float32, and float64; "
      f"got {score_dtype}"
    )
  for score in scores:
    if score.dtype != score_dtype:
      raise TypeError(
        "graph runtime score tensors must share a dtype before promotion, "
        f"got {score_dtype} and {score.dtype}"
      )
  if score_dtype in _LOW_PRECISION_DTYPES:
    return torch.float32
  return score_dtype


def _without_autocast(score: Tensor) -> ContextManager[object]:
  if score.device.type in {"cpu", "cuda"}:
    return cast(ContextManager[object], torch.autocast(device_type=score.device.type, enabled=False))
  return nullcontext()


def _promote(score: Tensor, compute_dtype: torch.dtype) -> Tensor:
  return score if score.dtype == compute_dtype else score.to(compute_dtype)


def _promote_uniforms(
  uniforms: Tensor,
  compute_dtype: torch.dtype,
) -> Tensor:
  if uniforms.dtype not in _GRAPH_DTYPES:
    raise TypeError(
      "graph sampling uniforms must be floating point, got "
      f"{uniforms.dtype}"
    )
  return _promote(uniforms, compute_dtype)


def alignment_log_partition(
  emission_scores: Tensor,
  transition_scores: Tensor,
  node_lengths: Tensor,
  target_lengths: Tensor,
  *,
  backend: GraphBackend = "auto",
) -> Tensor:
  """Dispatch the fixed-target alignment partition.

  FP16/BF16 model scores are promoted to FP32 DP.
  """
  compute_dtype = _graph_compute_dtype(emission_scores, transition_scores)
  selected = _resolve_backend(backend, emission_scores)
  with _without_autocast(emission_scores):
    emissions = _promote(emission_scores, compute_dtype)
    transitions = _promote(transition_scores, compute_dtype)
    if selected == "cuda":
      from .backends import cuda

      return cuda.alignment_log_partition(
        emissions, transitions, node_lengths, target_lengths)
    return _pytorch_alignment_log_partition(
      emissions, transitions, node_lengths, target_lengths)


def alignment_log_posterior(
  emission_scores: Tensor,
  transition_scores: Tensor,
  node_lengths: Tensor,
  target_lengths: Tensor,
  *,
  backend: GraphBackend = "auto",
) -> Tensor:
  """Dispatch the alignment posterior.

  Arbitrary posterior VJPs are preserved through promotion.
  """
  compute_dtype = _graph_compute_dtype(emission_scores, transition_scores)
  selected = _resolve_backend(backend, emission_scores)
  with _without_autocast(emission_scores):
    emissions = _promote(emission_scores, compute_dtype)
    transitions = _promote(transition_scores, compute_dtype)
    if selected == "cuda":
      from .backends import cuda

      return cuda.alignment_log_posterior(
        emissions, transitions, node_lengths, target_lengths)
    return _pytorch_alignment_log_posterior(
      emissions, transitions, node_lengths, target_lengths)


def path_log_partition(
  transition_scores: Tensor,
  node_lengths: Tensor,
  *,
  backend: GraphBackend = "auto",
) -> Tensor:
  """Dispatch the all-paths partition.

  FP16/BF16 transition scores are promoted to FP32 DP.
  """
  compute_dtype = _graph_compute_dtype(transition_scores)
  selected = _resolve_backend(backend, transition_scores)
  with _without_autocast(transition_scores):
    transitions = _promote(transition_scores, compute_dtype)
    if selected == "cuda":
      from .backends import cuda

      return cuda.path_log_partition(transitions, node_lengths)
    return _pytorch_path_log_partition(transitions, node_lengths)


@torch.no_grad()
def path_suffix_log_partition(
  transition_scores: Tensor,
  node_lengths: Tensor,
  *,
  backend: GraphBackend = "auto",
) -> Tensor:
  """Dispatch inference-only all-paths suffix weights.

  FP16/BF16 inputs are promoted to FP32 with autocast disabled. CUDA inputs
  use the compiled reverse DP by default; the explicit PyTorch backend keeps
  the reference on the input device.
  """
  compute_dtype = _graph_compute_dtype(transition_scores)
  selected = _resolve_backend(backend, transition_scores)
  with _without_autocast(transition_scores):
    transitions = _promote(transition_scores, compute_dtype)
    if selected == "cuda":
      from .backends import cuda

      return cuda.path_suffix_log_partition(transitions, node_lengths)
    return _pytorch_path_suffix_log_partition(transitions, node_lengths)


@torch.no_grad()
def viterbi_alignment(
  emission_scores: Tensor,
  transition_scores: Tensor,
  node_lengths: Tensor,
  target_lengths: Tensor,
  *,
  backend: GraphBackend = "auto",
) -> ViterbiAlignment:
  """Dispatch fixed-length Viterbi alignment in the graph-DP compute dtype."""
  compute_dtype = _graph_compute_dtype(emission_scores, transition_scores)
  selected = _resolve_backend(backend, emission_scores)
  with _without_autocast(emission_scores):
    emissions = _promote(emission_scores, compute_dtype)
    transitions = _promote(transition_scores, compute_dtype)
    if selected == "cuda":
      from .backends import cuda

      return cuda.viterbi_alignment(
        emissions, transitions, node_lengths, target_lengths)
    return _pytorch_viterbi_alignment(
      emissions, transitions, node_lengths, target_lengths)


@torch.no_grad()
def viterbi_path(
  node_scores: Tensor,
  transition_scores: Tensor,
  node_lengths: Tensor,
  *,
  backend: GraphBackend = "auto",
) -> ViterbiPath:
  """Dispatch maximum-score path decoding in the graph-DP compute dtype."""
  compute_dtype = _graph_compute_dtype(node_scores, transition_scores)
  selected = _resolve_backend(backend, node_scores)
  with _without_autocast(node_scores):
    nodes = _promote(node_scores, compute_dtype)
    transitions = _promote(transition_scores, compute_dtype)
    if selected == "cuda":
      from .backends import cuda

      return cuda.viterbi_path(nodes, transitions, node_lengths)
    return _pytorch_viterbi_path(nodes, transitions, node_lengths)


@torch.no_grad()
def sample_path(
  transition_scores: Tensor,
  node_lengths: Tensor,
  uniforms: Tensor,
  *,
  backend: GraphBackend = "auto",
) -> SampledPath:
  """Dispatch path sampling.

  Callers may keep externally generated uniforms in FP32.
  """
  compute_dtype = _graph_compute_dtype(transition_scores)
  selected = _resolve_backend(backend, transition_scores)
  with _without_autocast(transition_scores):
    transitions = _promote(transition_scores, compute_dtype)
    draws = _promote_uniforms(uniforms, compute_dtype)
    if selected == "cuda":
      from .backends import cuda

      return cuda.sample_path(transitions, node_lengths, draws)
    return _pytorch_sample_path(transitions, node_lengths, draws)


@torch.no_grad()
def sample_alignment(
  emission_scores: Tensor,
  transition_scores: Tensor,
  node_lengths: Tensor,
  target_lengths: Tensor,
  uniforms: Tensor,
  *,
  backend: GraphBackend = "auto",
) -> SampledAlignment:
  """Dispatch fixed-target alignment sampling.

  Low-precision scores and uniforms execute in FP32.
  """
  compute_dtype = _graph_compute_dtype(emission_scores, transition_scores)
  selected = _resolve_backend(backend, emission_scores)
  with _without_autocast(emission_scores):
    emissions = _promote(emission_scores, compute_dtype)
    transitions = _promote(transition_scores, compute_dtype)
    draws = _promote_uniforms(uniforms, compute_dtype)
    if selected == "cuda":
      from .backends import cuda

      return cuda.sample_alignment(
        emissions, transitions, node_lengths, target_lengths, draws)
    return _pytorch_sample_alignment(
      emissions, transitions, node_lengths, target_lengths, draws)


__all__ = [
  "GraphBackend",
  "alignment_log_partition",
  "alignment_log_posterior",
  "path_log_partition",
  "path_suffix_log_partition",
  "sample_alignment",
  "sample_path",
  "validate_backend",
  "viterbi_alignment",
  "viterbi_path",
]
