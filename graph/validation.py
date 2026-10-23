"""Shared validation and masking policy for dense DAG references.

Graph dynamic programs call only the metadata validators.  They check tensor
shape, dtype, device, and the small length vectors, without scanning complete
score tensors.  The public ``strict_validate_*`` helpers add full value and
topology checks for tests, debugging, and new component integration.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor


def _validate_transition_structure(transition_scores: Tensor) -> tuple[int, int]:
  if transition_scores.ndim != 3:
    raise ValueError(
      "transition_scores must have shape [batch, source, destination], "
      f"got {tuple(transition_scores.shape)}"
    )
  batch_size, source_count, destination_count = transition_scores.shape
  if source_count != destination_count:
    raise ValueError(
      "transition_scores must be square in its node dimensions, "
      f"got {tuple(transition_scores.shape)}"
    )
  if batch_size == 0:
    raise ValueError("empty batches are not supported")
  if transition_scores.dtype not in {torch.float32, torch.float64}:
    raise TypeError(
      "transition_scores supports torch.float32 and torch.float64 only, "
      f"got {transition_scores.dtype}"
    )
  return batch_size, source_count


def _validate_length_metadata(
  name: str,
  lengths: Tensor,
  batch_size: int,
  device: torch.device,
) -> None:
  if lengths.ndim != 1 or lengths.shape[0] != batch_size:
    raise ValueError(
      f"{name} must have shape [{batch_size}], got {tuple(lengths.shape)}"
    )
  if lengths.dtype != torch.long:
    raise TypeError(f"{name} must use torch.long, got {lengths.dtype}")
  if lengths.device != device:
    raise ValueError(f"{name} must be on {device}, got {lengths.device}")


def _validate_length_values(
  node_lengths: Tensor,
  max_node_count: int,
  target_lengths: Tensor | None = None,
  max_target_length: int | None = None,
) -> None:
  invalid = (node_lengths < 2) | (node_lengths > max_node_count)
  if target_lengths is not None:
    assert max_target_length is not None
    invalid = (
      invalid
      | (target_lengths < 2)
      | (target_lengths > max_target_length)
      | (target_lengths > node_lengths)
    )
  if not torch.any(invalid):
    return

  # Detailed diagnosis runs only after the single hot-path reduction failed.
  if torch.any(node_lengths < 2):
    raise ValueError("every node length must be at least 2")
  if torch.any(node_lengths > max_node_count):
    raise ValueError(
      f"node_lengths cannot exceed the padded node dimension {max_node_count}"
    )
  if target_lengths is None:
    return
  if torch.any(target_lengths < 2):
    raise ValueError("every target length must be at least 2")
  if torch.any(target_lengths > cast(int, max_target_length)):
    raise ValueError(
      "target_lengths cannot exceed the padded target dimension "
      f"{max_target_length}"
    )
  if torch.any(target_lengths > node_lengths):
    raise ValueError("target_lengths cannot exceed node_lengths")


def validate_transition_metadata(
  transition_scores: Tensor,
  node_lengths: Tensor,
) -> tuple[int, int]:
  """Validate transition metadata with one length-value reduction."""
  batch_size, max_node_count = _validate_transition_structure(transition_scores)
  _validate_length_metadata(
    "node_lengths",
    node_lengths,
    batch_size,
    transition_scores.device,
  )
  _validate_length_values(node_lengths, max_node_count)
  return batch_size, max_node_count


def validate_scored_path_metadata(
  node_scores: Tensor,
  transition_scores: Tensor,
  node_lengths: Tensor,
) -> tuple[int, int]:
  """Validate node/transition metadata for path-scoring algorithms."""
  batch_size, max_node_count = validate_transition_metadata(
    transition_scores,
    node_lengths,
  )
  expected = (batch_size, max_node_count)
  if tuple(node_scores.shape) != expected:
    raise ValueError(
      "node_scores shape must match transition batch/node dimensions, "
      f"expected {expected}, got {tuple(node_scores.shape)}"
    )
  if node_scores.dtype != transition_scores.dtype:
    raise TypeError(
      "node_scores and transition_scores must have the same dtype, "
      f"got {node_scores.dtype} and {transition_scores.dtype}"
    )
  if node_scores.device != transition_scores.device:
    raise ValueError(
      "node_scores and transition_scores must be on the same device, "
      f"got {node_scores.device} and {transition_scores.device}"
    )
  return batch_size, max_node_count


def validate_sample_path_metadata(
  transition_scores: Tensor,
  node_lengths: Tensor,
  uniforms: Tensor,
) -> tuple[int, int]:
  """Validate transition and pre-generated-uniform tensor metadata."""
  batch_size, max_node_count = validate_transition_metadata(
    transition_scores,
    node_lengths,
  )
  expected = (batch_size, max_node_count)
  if tuple(uniforms.shape) != expected:
    raise ValueError(
      "uniforms shape must match transition batch/node dimensions, "
      f"expected {expected}, got {tuple(uniforms.shape)}"
    )
  if uniforms.dtype != transition_scores.dtype:
    raise TypeError(
      "uniforms and transition_scores must have the same dtype, "
      f"got {uniforms.dtype} and {transition_scores.dtype}"
    )
  if uniforms.device != transition_scores.device:
    raise ValueError(
      "uniforms and transition_scores must be on the same device, "
      f"got {uniforms.device} and {transition_scores.device}"
    )
  return batch_size, max_node_count


def validate_sample_alignment_metadata(
  emission_scores: Tensor,
  transition_scores: Tensor,
  node_lengths: Tensor,
  target_lengths: Tensor,
  uniforms: Tensor,
) -> tuple[int, int, int]:
  """Validate fixed-target alignment and uniform tensor metadata."""
  batch_size, max_target_length, max_node_count = (
    validate_alignment_metadata(
      emission_scores,
      transition_scores,
      node_lengths,
      target_lengths,
    )
  )
  expected = (batch_size, max_target_length)
  if tuple(uniforms.shape) != expected:
    raise ValueError(
      "uniforms shape must match emission batch/target dimensions, "
      f"expected {expected}, got {tuple(uniforms.shape)}"
    )
  if uniforms.dtype != emission_scores.dtype:
    raise TypeError(
      "uniforms and score tensors must have the same dtype, "
      f"got {uniforms.dtype} and {emission_scores.dtype}"
    )
  if uniforms.device != emission_scores.device:
    raise ValueError(
      "uniforms and score tensors must be on the same device, "
      f"got {uniforms.device} and {emission_scores.device}"
    )
  return batch_size, max_target_length, max_node_count


def validate_alignment_metadata(
  emission_scores: Tensor,
  transition_scores: Tensor,
  node_lengths: Tensor,
  target_lengths: Tensor,
) -> tuple[int, int, int]:
  """Validate alignment metadata with one combined length-value reduction."""
  batch_size, max_node_count = _validate_transition_structure(transition_scores)
  if emission_scores.ndim != 3:
    raise ValueError(
      "emission_scores must have shape [batch, target, node], "
      f"got {tuple(emission_scores.shape)}"
    )
  emission_batch, max_target_length, emission_node_count = emission_scores.shape
  expected = (batch_size, max_target_length, max_node_count)
  if emission_batch != batch_size or emission_node_count != max_node_count:
    raise ValueError(
      "emission_scores shape must match transition batch/node dimensions, "
      f"expected {expected}, got {tuple(emission_scores.shape)}"
    )
  if emission_scores.dtype != transition_scores.dtype:
    raise TypeError(
      "emission_scores and transition_scores must have the same dtype, "
      f"got {emission_scores.dtype} and {transition_scores.dtype}"
    )
  if emission_scores.device != transition_scores.device:
    raise ValueError(
      "emission_scores and transition_scores must be on the same device, "
      f"got {emission_scores.device} and {transition_scores.device}"
    )
  _validate_length_metadata(
    "node_lengths",
    node_lengths,
    batch_size,
    transition_scores.device,
  )
  _validate_length_metadata(
    "target_lengths",
    target_lengths,
    batch_size,
    transition_scores.device,
  )
  _validate_length_values(
    node_lengths,
    max_node_count,
    target_lengths,
    max_target_length,
  )
  return batch_size, max_target_length, max_node_count


def mask_dense_dag(
  transition_scores: Tensor,
  node_lengths: Tensor,
) -> tuple[Tensor, Tensor]:
  """Mask padding and entries outside the strict-forward DAG topology."""
  max_node_count = transition_scores.shape[1]
  node_index = torch.arange(max_node_count, device=transition_scores.device)
  valid_nodes = node_index.unsqueeze(0) < node_lengths.unsqueeze(1)
  strict_forward = node_index.unsqueeze(1) < node_index.unsqueeze(0)
  valid_edges = (
    valid_nodes.unsqueeze(2)
    & valid_nodes.unsqueeze(1)
    & strict_forward.unsqueeze(0)
  )
  return transition_scores.masked_fill(~valid_edges, -torch.inf), valid_nodes


def _strict_validate_transition_values(
  transition_scores: Tensor,
  node_lengths: Tensor,
) -> None:
  max_node_count = transition_scores.shape[1]
  node_index = torch.arange(max_node_count, device=transition_scores.device)
  valid_nodes = node_index.unsqueeze(0) < node_lengths.unsqueeze(1)
  valid_pairs = valid_nodes.unsqueeze(2) & valid_nodes.unsqueeze(1)
  values = transition_scores[valid_pairs]
  if (torch.isnan(values) | torch.isposinf(values)).any():
    raise ValueError("valid transition scores cannot contain NaN or +inf")

  non_forward = node_index.unsqueeze(1) >= node_index.unsqueeze(0)
  invalid_edges = valid_pairs & non_forward.unsqueeze(0)
  if not torch.isneginf(transition_scores[invalid_edges]).all():
    raise ValueError(
      "valid diagonal and backward transitions must be -inf; nodes must be "
      "provided in a strict topological order"
    )


def strict_validate_transition_inputs(
  transition_scores: Tensor,
  node_lengths: Tensor,
) -> None:
  """Run metadata plus full transition value/topology validation."""
  validate_transition_metadata(transition_scores, node_lengths)
  _strict_validate_transition_values(transition_scores, node_lengths)


def strict_validate_scored_path_inputs(
  node_scores: Tensor,
  transition_scores: Tensor,
  node_lengths: Tensor,
) -> None:
  """Run metadata plus full node/transition value validation."""
  _, max_node_count = validate_scored_path_metadata(
    node_scores,
    transition_scores,
    node_lengths,
  )
  node_index = torch.arange(max_node_count, device=node_scores.device)
  valid_nodes = node_index.unsqueeze(0) < node_lengths.unsqueeze(1)
  values = node_scores[valid_nodes]
  if (torch.isnan(values) | torch.isposinf(values)).any():
    raise ValueError("valid node scores cannot contain NaN or +inf")
  _strict_validate_transition_values(transition_scores, node_lengths)


def strict_validate_sample_path_inputs(
  transition_scores: Tensor,
  node_lengths: Tensor,
  uniforms: Tensor,
) -> None:
  """Run metadata plus full transition/uniform value validation."""
  _, max_node_count = validate_sample_path_metadata(
    transition_scores,
    node_lengths,
    uniforms,
  )
  reverse_step = torch.arange(max_node_count, device=uniforms.device)
  potentially_used = (
    reverse_step.unsqueeze(0) < (node_lengths - 1).unsqueeze(1)
  )
  values = uniforms[potentially_used]
  if (
    ~torch.isfinite(values)
    | (values < 0)
    | (values >= 1)
  ).any():
    raise ValueError("potentially used uniforms must be finite and in [0, 1)")
  _strict_validate_transition_values(transition_scores, node_lengths)


def strict_validate_sample_alignment_inputs(
  emission_scores: Tensor,
  transition_scores: Tensor,
  node_lengths: Tensor,
  target_lengths: Tensor,
  uniforms: Tensor,
) -> None:
  """Run full fixed-target alignment and uniform value validation."""
  _, max_target_length, _ = validate_sample_alignment_metadata(
    emission_scores,
    transition_scores,
    node_lengths,
    target_lengths,
    uniforms,
  )
  reverse_step = torch.arange(max_target_length, device=uniforms.device)
  potentially_used = (
    reverse_step.unsqueeze(0) < (target_lengths - 1).unsqueeze(1)
  )
  values = uniforms[potentially_used]
  if (
    ~torch.isfinite(values)
    | (values < 0)
    | (values >= 1)
  ).any():
    raise ValueError("potentially used uniforms must be finite and in [0, 1)")
  _strict_validate_alignment_values(
    emission_scores,
    transition_scores,
    node_lengths,
    target_lengths,
  )


def strict_validate_alignment_inputs(
  emission_scores: Tensor,
  transition_scores: Tensor,
  node_lengths: Tensor,
  target_lengths: Tensor,
) -> None:
  """Run metadata plus full emission/transition value validation."""
  validate_alignment_metadata(
    emission_scores,
    transition_scores,
    node_lengths,
    target_lengths,
  )
  _strict_validate_alignment_values(
    emission_scores,
    transition_scores,
    node_lengths,
    target_lengths,
  )


def _strict_validate_alignment_values(
  emission_scores: Tensor,
  transition_scores: Tensor,
  node_lengths: Tensor,
  target_lengths: Tensor,
) -> None:
  """Scan alignment score values after metadata has been validated."""
  _, max_target_length, max_node_count = emission_scores.shape
  target_index = torch.arange(
    max_target_length, device=emission_scores.device)
  node_index = torch.arange(max_node_count, device=emission_scores.device)
  valid_targets = target_index.unsqueeze(0) < target_lengths.unsqueeze(1)
  valid_nodes = node_index.unsqueeze(0) < node_lengths.unsqueeze(1)
  values = emission_scores[
    valid_targets.unsqueeze(2) & valid_nodes.unsqueeze(1)
  ]
  if (torch.isnan(values) | torch.isposinf(values)).any():
    raise ValueError("valid emission scores cannot contain NaN or +inf")
  _strict_validate_transition_values(transition_scores, node_lengths)


__all__ = [
  "strict_validate_alignment_inputs",
  "strict_validate_sample_path_inputs",
  "strict_validate_sample_alignment_inputs",
  "strict_validate_scored_path_inputs",
  "strict_validate_transition_inputs",
]
