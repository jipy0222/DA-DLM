"""Reveal sampled content on a selected DADLM path.

The caller owns path decoding, emission sampling, graph invalidation, and stop
handling.  This module only decides which new path tokens become anchors.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

from .config import DiffusionConfig
from .reveal import _correlated_uniform, _infill_keep, _rounded_quota, _uniform


def _path_confidence(
    path: Tensor,
    emission: Tensor,
    transitions: Tensor,
    weight: float,
    *,
    crf: bool,
    suffix: Tensor | None,
) -> Tensor:
    """Probability-domain emission/adjacent-edge confidence for content."""
    if weight == 0:
        return emission
    sources = path[:-1]
    targets = path[1:]
    log_q = transitions[sources, targets].float()
    if crf:
        assert suffix is not None
        source_beta = suffix[sources]
        target_beta = suffix[targets]
        log_q = log_q + target_beta - source_beta
    if not bool(torch.isfinite(log_q).all()):
        if crf:
            assert suffix is not None
            if not bool(torch.isfinite(suffix[sources]).all() & torch.isfinite(suffix[targets]).all()):
                raise ValueError("selected path has an unreachable CRF suffix")
        raise ValueError("selected path contains a nonfinite transition confidence")
    log_q = log_q.clamp_max(0)
    # Average in log space so the product of tiny edge probabilities cannot underflow.
    adjacent = ((log_q[:-1] + log_q[1:]) / 2).exp().to(emission.dtype)
    return (1 - weight) * emission + weight * adjacent


def _nearby_keep(
    old_path: Tensor,
    mask_token_id: int,
    remask_probability: float,
    config: DiffusionConfig,
    generator: torch.Generator | None,
) -> Tensor:
    """Use selected-path coordinates, with BOS but without EOS influence."""
    path_length = old_path.numel()
    new = old_path.eq(mask_token_id)
    new[0] = False
    new[-1] = False
    keep = torch.zeros(path_length, device=old_path.device, dtype=torch.bool)
    if not bool(new.any()):
        return keep

    accepted = (~old_path.eq(mask_token_id)).to(torch.float64)
    accepted[-1] = 0
    left = torch.zeros_like(accepted)
    right = torch.zeros_like(accepted)
    decay = config.nearby_decay
    if decay == 0:
        left[1:] = accepted[:-1]
        right[:-1] = accepted[1:]
    else:
        for i in range(1, path_length):
            left[i] = accepted[i - 1] + decay * left[i - 1]
        for i in range(path_length - 2, -1, -1):
            right[i] = accepted[i + 1] + decay * right[i + 1]

    log_weight = -config.nearby_unmask_lambda * (left + right) / 2
    # A common offset cancels in weight / mean(weight).  Shifting before exp
    # keeps even very large lambda values finite without changing the rule.
    relative = (log_weight - log_weight[new].max()).exp()
    mean = relative[new].mean()
    adjusted = (remask_probability * relative / mean).clamp(0, 1)
    uniform = _correlated_uniform(
        torch.Size((1, path_length)), config.remask_rho, old_path.device, generator
    )[0]
    keep[new] = uniform[new] >= adjusted[new]
    return keep


def _physical_reveal_draws(
    current_ids: Tensor, mask_token_id: int, remask_probability: float,
    config: DiffusionConfig, generator: torch.Generator | None,
) -> tuple[Tensor | None, Tensor | None]:
    batch, width = current_ids.shape
    physical_u = None
    # Physical-coordinate draws preserve the grouped consecutive rule: the
    # count and correlated ordering include viable MASK nodes off this path.
    physical_new = current_ids.eq(mask_token_id)
    physical_new[:, 0] = False
    physical_new[:, -1] = False
    if config.strategy == "consecutive" and config.remask_rho > 0:
        count_u = _uniform(current_ids.shape, current_ids.device, generator)
        remask_count = ((count_u < remask_probability) & physical_new).sum(dim=1)
        correlated = _correlated_uniform(
            current_ids.shape, config.remask_rho, current_ids.device, generator
        ).masked_fill(~physical_new, float("inf"))
        order = correlated.argsort(dim=1, stable=True)
        rank = torch.empty_like(order)
        rank.scatter_(1, order, torch.arange(width, device=current_ids.device).expand(batch, -1))
        physical_keep = physical_new & (rank >= remask_count[:, None])
    elif config.strategy in {"default", "consecutive"}:
        physical_u = _uniform(current_ids.shape, current_ids.device, generator)
        physical_keep = physical_new & (physical_u >= remask_probability)
    elif config.strategy == "first_hitting":
        physical_u = _uniform(current_ids.shape, current_ids.device, generator)
        physical_keep = None
    else:
        physical_u = None
        physical_keep = None

    return physical_u, physical_keep


def _path_reveal_mask(
    path: Tensor, current_ids: Tensor, new: Tensor, emission: Tensor, transitions: Tensor, *,
    mask_token_id: int, step: int, config: DiffusionConfig, transition_weight: float,
    crf: bool, suffix_weights: Tensor | None, remask_probability: float,
    physical_u: Tensor | None, physical_keep: Tensor | None, generator: torch.Generator | None,
) -> Tensor:
    content = path[1:-1]
    new_keep: int | None = None
    # Compare the context fraction without rounding the configured threshold.
    infill = _infill_keep(new[None], emission[None], config, context_dtype=torch.float64)[0]
    length = content.numel()
    if config.strategy in {"confidence", "sub_block"}:
        accepted = int((~new).sum())
        quota = _rounded_quota(length, step + 1, config.num_steps)
        new_keep = max(quota - accepted, 0)

    if config.strategy == "first_hitting":
        eligible = new & ~infill
        ordinary = torch.zeros_like(new)
        if bool(eligible.any()):
            assert physical_u is not None
            scores = physical_u[content].masked_fill(~eligible, 2)
            ordinary[int(scores.argmin())] = True
    elif config.strategy in {"confidence", "confidence_threshold"}:
        confidence = _path_confidence(
            path, emission, transitions, transition_weight,
            crf=crf, suffix=suffix_weights,
        )
        if config.strategy == "confidence":
            uniform = _uniform(torch.Size((length,)), current_ids.device, generator).clamp(1e-6, 1 - 1e-6)
            gumbel = -torch.log(-torch.log(uniform))
            scores = confidence.float() + config.confidence_temperature * gumbel
            scores = scores.masked_fill(~new, float("-inf"))
            order = scores.argsort(descending=True, stable=True)
            ordinary = torch.zeros_like(new)
            ordinary[order[:cast(int, new_keep)]] = True
        else:
            high = new & (confidence > config.confidence_threshold)
            if bool(high.any()):
                ordinary = high
            else:
                scores = confidence.masked_fill(~new, float("-inf"))
                order = scores.argsort(descending=True, stable=True)
                ordinary = torch.zeros_like(new)
                ordinary[order[: min(config.fallback_k, int(new.sum()))]] = True
    elif config.strategy == "sub_block":
        ordinary = torch.zeros_like(new)
        ordinary[new.nonzero(as_tuple=True)[0][:cast(int, new_keep)]] = True
    elif config.strategy == "nearby":
        ordinary = _nearby_keep(
            current_ids[path], mask_token_id, remask_probability, config, generator
        )[1:-1]
    else:
        assert physical_keep is not None
        ordinary = physical_keep[content]

    accept = new & (ordinary | infill)
    return accept


def reveal_dag_tokens(
    current_ids: Tensor,
    candidates: Tensor,
    path_nodes: Tensor,
    path_lengths: Tensor,
    emission_confidence: Tensor,
    transitions: Tensor,
    *,
    mask_token_id: int,
    pad_token_id: int,
    step: int,
    config: DiffusionConfig,
    transition_weight: float,
    crf: bool = False,
    suffix_weights: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Accept new sampled tokens on ragged paths; preserve everything else.

    ``emission_confidence[b, i]`` is the filtered probability of the sampled
    token at physical node ``i``. ``transitions`` contains local log
    probabilities or CRF raw log potentials. ``suffix_weights`` may cache the
    full-graph CRF beta from the *pre-update* graph.
    """
    if current_ids.ndim != 2 or candidates.shape != current_ids.shape:
        raise ValueError("current_ids and candidates must have shape [B, N]")
    batch, width = current_ids.shape
    if (path_nodes.ndim != 2 or path_nodes.shape[0] != batch
            or path_lengths.shape != (batch,)
            or emission_confidence.shape != current_ids.shape
            or transitions.shape != (batch, width, width)):
        raise ValueError("path, confidence, or transition shape mismatch")
    if not 0 <= step < config.num_steps:
        raise ValueError("step is outside the configured update range")
    if not 0 <= transition_weight <= 1:
        raise ValueError("transition_weight must be in [0, 1]")
    if suffix_weights is not None and suffix_weights.shape != current_ids.shape:
        raise ValueError("suffix_weights must have shape [B, N]")

    needs_confidence = config.strategy in {"confidence", "confidence_threshold"}
    if crf and needs_confidence and transition_weight > 0 and suffix_weights is None:
        from graph.runtime import path_suffix_log_partition

        node_lengths = torch.full(
            (batch,), width, device=current_ids.device, dtype=torch.long
        )
        suffix_weights = path_suffix_log_partition(transitions, node_lengths)

    result = current_ids.clone()
    remask_probability = (config.num_steps - step - 1) / (config.num_steps - step)

    physical_u, physical_keep = _physical_reveal_draws(
        current_ids, mask_token_id, remask_probability, config, generator,
    )

    for b, path_length in enumerate(path_lengths.tolist()):
        if not 2 <= path_length <= path_nodes.shape[1]:
            raise ValueError("each path must contain both internal endpoints")
        path = path_nodes[b, :path_length].long()
        if (int(path[0]) != 0 or int(path[-1]) != width - 1
                or bool((path[1:] <= path[:-1]).any())):
            raise ValueError("path must strictly increase from BOS to EOS")
        content = path[1:-1]
        if not content.numel():
            continue
        old_content_ids = current_ids[b, content]
        if bool(old_content_ids.eq(pad_token_id).any()):
            raise ValueError("selected path contains inactive PAD content")
        new = old_content_ids.eq(mask_token_id)
        if not bool(new.any()):
            continue
        proposed = candidates[b, content]
        if bool((proposed[new].eq(mask_token_id) | proposed[new].eq(pad_token_id)).any()):
            raise ValueError("selected new content needs a sampled non-MASK, non-PAD token")
        emission = emission_confidence[b, content]
        if not bool(torch.isfinite(emission[new]).all() &
                    (emission[new] >= 0).all() & (emission[new] <= 1).all()):
            raise ValueError("selected emission confidence must be finite in [0, 1]")

        accept = _path_reveal_mask(
            path, current_ids[b], new, emission, transitions[b], mask_token_id=mask_token_id,
            step=step, config=config, transition_weight=transition_weight, crf=crf,
            suffix_weights=None if suffix_weights is None else suffix_weights[b],
            remask_probability=remask_probability, physical_u=None if physical_u is None else physical_u[b],
            physical_keep=None if physical_keep is None else physical_keep[b], generator=generator,
        )
        result[b, content] = torch.where(accept, proposed, old_content_ids)
    return result


__all__ = ["reveal_dag_tokens"]
