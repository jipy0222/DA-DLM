"""Position-reveal rules for one absorbing-mask diffusion block."""

from __future__ import annotations

import math

import torch

from .config import DiffusionConfig


def _uniform(shape: torch.Size, device: torch.device, generator: torch.Generator | None) -> torch.Tensor:
    return torch.rand(shape, device=device, dtype=torch.float32, generator=generator)


def _correlated_uniform(
    shape: torch.Size, rho: float, device: torch.device, generator: torch.Generator | None
) -> torch.Tensor:
    if rho <= 0:
        return _uniform(shape, device, generator)
    noise = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
    gaussian = torch.empty_like(noise)
    gaussian[:, 0] = noise[:, 0]
    scale = math.sqrt(1 - rho * rho)
    for position in range(1, shape[1]):
        gaussian[:, position] = rho * gaussian[:, position - 1] + scale * noise[:, position]
    return (0.5 * (1 + torch.erf(gaussian / math.sqrt(2)))).float()


def _rounded_quota(length: int, completed_steps: int, total_steps: int) -> int:
    whole, remainder = divmod(length * completed_steps, total_steps)
    if 2 * remainder > total_steps or (2 * remainder == total_steps and whole % 2):
        whole += 1
    return whole


def _infill_positions(is_new: torch.Tensor, k: int) -> torch.Tensor:
    """Find short gaps with a virtual left anchor and a real right anchor."""
    length = is_new.shape[1]
    positions = torch.arange(length, device=is_new.device)[None]
    left = torch.where(~is_new, positions, -1).cummax(dim=1).values
    right = torch.where(~is_new, positions, length).flip(1).cummin(dim=1).values.flip(1)
    return is_new & (right < length) & (right - left - 1 <= k)


def _infill_keep(
    is_new: torch.Tensor, confidence: torch.Tensor, config: DiffusionConfig,
    *, context_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if config.infilling_k == 0:
        return torch.zeros_like(is_new)
    context = (~is_new).float().mean(dim=-1).to(context_dtype)
    keep = _infill_positions(is_new, config.infilling_k)
    keep &= (context >= config.infilling_min_context).unsqueeze(-1)
    if config.infilling_threshold <= 0 or not keep.any():
        return keep

    starts = keep & ~torch.nn.functional.pad(keep[:, :-1], (1, 0), value=False)
    segment_id = starts.cumsum(dim=-1) * keep
    segment_count = int(segment_id.max().item())
    segment_min = torch.ones(
        (keep.shape[0], segment_count + 1), device=keep.device, dtype=confidence.dtype
    )
    values = torch.where(keep, confidence, torch.ones_like(confidence))
    segment_min.scatter_reduce_(1, segment_id, values, reduce="amin", include_self=False)
    return keep & (segment_min.gather(1, segment_id) >= config.infilling_threshold)


def _confidence_remask(
    is_new: torch.Tensor,
    confidence: torch.Tensor,
    quota: int,
    temperature: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    batch, length = is_new.shape
    committed_count = (~is_new).sum(dim=-1, keepdim=True)
    new_keep = (quota - committed_count).clamp_min(0)
    remask_count = (is_new.sum(dim=-1, keepdim=True) - new_keep).clamp_min(0)
    log_conf = confidence.clamp_min(1e-30).log().float()
    log_conf = log_conf.masked_fill(~is_new, float("inf"))
    uniform = _uniform(log_conf.shape, log_conf.device, generator).clamp(1e-6, 1 - 1e-6)
    score = log_conf - temperature * torch.log(-torch.log(uniform))
    score = score.masked_fill(~is_new, float("inf"))
    sorted_indices = score.argsort(dim=-1)
    rank = torch.empty_like(sorted_indices)
    rank.scatter_(1, sorted_indices, torch.arange(length, device=is_new.device).expand(batch, -1))
    return is_new & (rank < remask_count)


def _threshold_remask(
    is_new: torch.Tensor, confidence: torch.Tensor, threshold: float, fallback_k: int
) -> torch.Tensor:
    scores = confidence.masked_fill(~is_new, -1)
    high = scores > threshold
    sorted_indices = scores.sort(dim=-1, descending=True).indices
    rank = torch.arange(is_new.shape[1], device=is_new.device).expand_as(sorted_indices)
    fallback = torch.zeros_like(is_new).scatter_(1, sorted_indices, rank < fallback_k) & is_new
    keep = torch.where(high.any(dim=-1, keepdim=True), high, fallback)
    return is_new & ~keep


def _nearby_remask(
    is_new: torch.Tensor,
    probability: float,
    config: DiffusionConfig,
    generator: torch.Generator | None,
) -> torch.Tensor:
    length = is_new.shape[1]
    fixed = (~is_new).float()
    left = torch.zeros_like(fixed)
    right = torch.zeros_like(fixed)
    if config.nearby_decay == 0:
        left[:, 1:] = fixed[:, :-1]
        right[:, :-1] = fixed[:, 1:]
    else:
        for position in range(1, length):
            left[:, position] = fixed[:, position - 1] + config.nearby_decay * left[:, position - 1]
        for position in range(length - 2, -1, -1):
            right[:, position] = fixed[:, position + 1] + config.nearby_decay * right[:, position + 1]
    new_count = is_new.sum(dim=-1, keepdim=True)
    score = (left + right) / 2
    offset = score.masked_fill(~is_new, torch.inf).amin(dim=-1, keepdim=True)
    offset = torch.where(new_count > 0, offset, 0.)
    # A shared offset cancels in weight / mean(weight); the largest weight is 1.
    relative = (score - offset).masked_fill(~is_new, 0.)
    weight = (-config.nearby_unmask_lambda * relative).exp().masked_fill(~is_new, 0.)
    mean_weight = weight.sum(dim=-1, keepdim=True) / new_count.clamp_min(1)
    mean_weight = torch.where(new_count > 0, mean_weight, torch.ones_like(mean_weight))
    adjusted = (probability * weight / mean_weight).clamp(0, 1)
    uniform = _correlated_uniform(is_new.shape, config.remask_rho, is_new.device, generator)
    return is_new & (uniform < adjusted)


def _consecutive_remask(
    is_new: torch.Tensor,
    probability: float,
    rho: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if rho <= 0:
        return is_new & (_uniform(is_new.shape, is_new.device, generator) < probability)
    count_uniform = _uniform(is_new.shape, is_new.device, generator)
    remask_count = ((count_uniform < probability) & is_new).sum(dim=-1, keepdim=True)
    uniform = _correlated_uniform(is_new.shape, rho, is_new.device, generator)
    uniform = torch.where(is_new, uniform, torch.ones_like(uniform))
    sorted_indices = uniform.argsort(dim=-1)
    rank = torch.empty_like(sorted_indices)
    rank.scatter_(1, sorted_indices, torch.arange(is_new.shape[1], device=is_new.device).expand_as(rank))
    return is_new & (rank < remask_count)


def reveal_tokens(
    current_ids: torch.Tensor,
    candidates: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    mask_token_id: int,
    step: int,
    config: DiffusionConfig,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply one reveal update; stopping and suffix invalidation are separate."""
    if current_ids.ndim != 2 or candidates.shape != current_ids.shape:
        raise ValueError("current_ids and candidates must both have shape [batch, block_length]")
    if probabilities.shape[:2] != current_ids.shape:
        raise ValueError("probabilities must have shape [batch, block_length, vocabulary]")
    if not 0 <= step < config.num_steps:
        raise ValueError("step is outside the configured update range")

    length = current_ids.shape[1]
    is_new = current_ids.eq(mask_token_id)
    confidence = probabilities.gather(-1, candidates.unsqueeze(-1)).squeeze(-1)
    infill_keep = _infill_keep(is_new, confidence, config)
    remask_probability = (config.num_steps - step - 1) / (config.num_steps - step)
    quota = _rounded_quota(length, step + 1, config.num_steps)

    if config.strategy == "first_hitting":
        eligible = is_new & ~infill_keep
        random_scores = _uniform(current_ids.shape, current_ids.device, generator)
        random_scores = random_scores.masked_fill(~eligible, 2)
        selected = random_scores.argmin(dim=-1, keepdim=True)
        ordinary_keep = torch.zeros_like(eligible).scatter_(1, selected, True) & eligible
        remask = eligible & ~ordinary_keep
    elif config.strategy == "confidence":
        remask = _confidence_remask(
            is_new, confidence, quota, config.confidence_temperature, generator
        )
    elif config.strategy == "confidence_threshold":
        remask = _threshold_remask(
            is_new, confidence, config.confidence_threshold, config.fallback_k
        )
    elif config.strategy == "sub_block":
        positions = torch.arange(length, device=current_ids.device).unsqueeze(0)
        remask = is_new & (positions >= quota)
    elif config.strategy == "nearby":
        remask = _nearby_remask(is_new, remask_probability, config, generator)
    elif config.strategy == "consecutive":
        remask = _consecutive_remask(is_new, remask_probability, config.remask_rho, generator)
    else:
        remask = is_new & (_uniform(current_ids.shape, current_ids.device, generator) < remask_probability)

    remask &= ~infill_keep
    proposal = torch.where(remask, mask_token_id, candidates)
    return torch.where(is_new, proposal, current_ids)
