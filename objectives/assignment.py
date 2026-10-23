"""Discrete target-to-node assignments for DADLM training blocks."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

from graph import runtime
from graph.runtime import GraphBackend

_GROUPED = frozenset({"canonical", "glat", "partial_glat", "sample", "partial_sample"})
_FREE = frozenset({"uniform", "random", "glat", "partial_glat", "sample", "partial_sample"})


def validate_assignment(name: str, grouped: bool) -> str:
    """Return a supported strategy name for the requested DAG topology."""
    if not isinstance(name, str) or name not in (_GROUPED if grouped else _FREE):
        topology = "grouped" if grouped else "free"
        raise ValueError(f"unsupported {topology} assignment: {name!r}")
    return name


def initial_assignment(
    lengths: Tensor,
    *,
    block_size: int,
    dag_up_ratio: int,
    strategy: str,
    fixed_bos: Tensor,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Place real targets into the fixed physical block, padding with ``-1``.

    Canonical draws one node uniformly within each original group. Free random
    draws an unordered subset of legal content nodes and then orders it; a fixed
    BOS reserves node 1 before that draw. Free uniform uses the integer lattice
    ``1 + floor(i * (N - 2) / L)`` for target index ``i``.
    """
    if strategy not in {"canonical", "uniform", "random"}:
        raise ValueError(
            f"initial assignment requires canonical, uniform, or random; got {strategy!r}"
        )
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    if not isinstance(dag_up_ratio, int) or dag_up_ratio < 1:
        raise ValueError("dag_up_ratio must be a positive integer")
    if lengths.ndim != 1 or lengths.dtype != torch.long:
        raise ValueError("lengths must be a one-dimensional long tensor")
    if fixed_bos.shape != lengths.shape or fixed_bos.dtype != torch.bool:
        raise ValueError("fixed_bos must be a bool tensor matching lengths")
    if fixed_bos.device != lengths.device:
        raise ValueError("fixed_bos and lengths must be on the same device")

    node_count = block_size * dag_up_ratio
    content_count = node_count - 2
    if content_count < 1:
        raise ValueError("the physical DAG has no content node")
    capacity = block_size if strategy == "canonical" else content_count
    if torch.any((lengths < 0) | (lengths > block_size) | (lengths > capacity)):
        raise ValueError("target length exceeds the block or its content-node capacity")
    if torch.any(fixed_bos & lengths.eq(0)):
        raise ValueError("fixed BOS requires a nonempty target")
    if strategy == "canonical" and dag_up_ratio < 2 and torch.any(lengths > 0):
        raise ValueError("canonical group zero has no legal content node")

    batch_size = lengths.shape[0]
    target_index = torch.arange(block_size, device=lengths.device)
    active = target_index.unsqueeze(0) < lengths.unsqueeze(1)

    if strategy == "canonical":
        lower = (target_index * dag_up_ratio).clamp_min(1)
        upper = ((target_index + 1) * dag_up_ratio - 1).clamp_max(node_count - 2)
        width = upper - lower + 1
        draws = torch.rand(
            (batch_size, block_size), device=lengths.device, generator=generator
        )
        positions = lower.unsqueeze(0) + (draws * width.unsqueeze(0)).long()
        positions[:, 0] = torch.where(
            fixed_bos, torch.ones_like(positions[:, 0]), positions[:, 0]
        )
    elif strategy == "uniform":
        positions = 1 + (
            target_index.unsqueeze(0)
            * content_count
            // lengths.clamp_min(1).unsqueeze(1)
        )
    else:
        draws = torch.rand(
            (batch_size, content_count), device=lengths.device, generator=generator
        )
        draws[:, 0] = torch.where(
            fixed_bos, torch.full_like(draws[:, 0], torch.inf), draws[:, 0]
        )
        ranked_nodes = draws.argsort(dim=1) + 1
        choice_index = (
            target_index.unsqueeze(0) - fixed_bos.long().unsqueeze(1)
        ).clamp(0, content_count - 1)
        positions = ranked_nodes.gather(1, choice_index)
        positions[:, 0] = torch.where(
            fixed_bos, torch.ones_like(positions[:, 0]), positions[:, 0]
        )
        positions = positions.masked_fill(~active, node_count).sort(dim=1).values

    return cast('Tensor', positions.masked_fill(~active, -1))


@torch.no_grad()
def aligned_assignment(
    emission_scores: Tensor,
    transition_scores: Tensor,
    target_lengths: Tensor,
    *,
    strategy: str,
    backend: GraphBackend = "auto",
    uniforms: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Select target-node placements from a fixed-target prepass alignment.

    ``target_lengths`` count real targets, excluding the two internal graph
    endpoints. Partial strategies use the caller's already-adjusted soft scores;
    this function does not alter hard ``-inf`` constraints.
    """
    if strategy not in {"glat", "partial_glat", "sample", "partial_sample"}:
        raise ValueError(
            f"aligned assignment requires GLAT or SAMPLE; got {strategy!r}"
        )
    if emission_scores.ndim != 3:
        raise ValueError("emission_scores must have shape [blocks, target, node]")
    batch_size, graph_target_width, node_count = emission_scores.shape
    if target_lengths.shape != (batch_size,) or target_lengths.dtype != torch.long:
        raise ValueError("target_lengths must be a long tensor with one entry per block")
    if target_lengths.device != emission_scores.device:
        raise ValueError("target_lengths and scores must be on the same device")
    if torch.any((target_lengths < 0) | (target_lengths > graph_target_width - 2)):
        raise ValueError("target_lengths exceed the supplied target capacity")
    bad_emissions = (
        torch.isnan(emission_scores) | torch.isposinf(emission_scores)
    ).flatten(1).any(dim=1)
    bad_transitions = (
        torch.isnan(transition_scores) | torch.isposinf(transition_scores)
    ).flatten(1).any(dim=1)
    bad_scores = bad_emissions | bad_transitions
    if torch.any(bad_scores):
        rows = bad_scores.nonzero(as_tuple=True)[0].tolist()
        raise ValueError(f"nonfinite prepass scores for blocks {rows}")

    node_lengths = torch.full_like(target_lengths, node_count)
    graph_target_lengths = target_lengths + 2
    if strategy in {"glat", "partial_glat"}:
        if uniforms is not None:
            raise ValueError("uniforms apply only to SAMPLE assignments")
        result = runtime.viterbi_alignment(
            emission_scores,
            transition_scores,
            node_lengths,
            graph_target_lengths,
            backend=backend,
        )
        if torch.any(~torch.isfinite(result.scores)):
            raise ValueError("prepass target alignment is unreachable or nonfinite")
    else:
        if uniforms is None:
            draw_dtype = (
                torch.float32
                if emission_scores.dtype in {torch.float16, torch.bfloat16}
                else emission_scores.dtype
            )
            uniforms = torch.rand(
                (batch_size, graph_target_width),
                device=emission_scores.device,
                dtype=draw_dtype,
                generator=generator,
            )
        result = runtime.sample_alignment(
            emission_scores,
            transition_scores,
            node_lengths,
            graph_target_lengths,
            uniforms,
            backend=backend,
        )
        if torch.any(result.nodes[:, 0].ne(0)):
            raise ValueError("prepass target alignment is unreachable")

    target_index = torch.arange(
        graph_target_width - 2, device=emission_scores.device
    )
    return result.nodes[:, 1:-1].masked_fill(
        target_index.unsqueeze(0) >= target_lengths.unsqueeze(1), -1
    )


__all__ = ["aligned_assignment", "initial_assignment", "validate_assignment"]
