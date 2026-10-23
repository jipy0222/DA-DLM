"""Request validation and model execution scope shared by generators."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from typing import Generator, Optional

import torch
from torch import nn

from models.types import NativeModel

from .config import SamplingConfig, StopConfig
from .types import GenerationResult, GenerationStats


def cache_dtype(model: NativeModel) -> torch.dtype:
    weight = model.token_embedding.weight
    device_type = weight.device.type
    if torch.is_autocast_enabled(device_type):
        return torch.get_autocast_dtype(device_type)
    return weight.dtype


@contextmanager
def generation_context(model: nn.Module) -> Generator[None, None, None]:
    modules = tuple(model.modules())
    training = tuple(module.training for module in modules)
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        for module, was_training in zip(modules, training):
            module.training = was_training


def prepare_request(
    model: NativeModel,
    prompt_ids: torch.Tensor,
    generation_length: int,
    start_token_id: Optional[int],
    sampling: SamplingConfig,
    stop: StopConfig,
    *,
    mask_token_id: Optional[int] = None,
) -> SamplingConfig:
    """Validate common inputs and add known non-output token IDs."""
    if not isinstance(prompt_ids, torch.Tensor) or prompt_ids.dtype != torch.long:
        raise TypeError("prompt_ids must be a long tensor")
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] == 0:
        raise ValueError("prompt_ids must have shape [nonempty batch, width]")
    if type(generation_length) is not int or generation_length <= 0:
        raise ValueError("generation_length must be a positive integer")
    if not isinstance(sampling, SamplingConfig) or not isinstance(stop, StopConfig):
        raise TypeError("sampling and stop must be their configuration types")

    parameter = next(model.parameters())
    if prompt_ids.device != parameter.device:
        raise ValueError("prompt_ids and model must share a device")
    vocab_size = model.config.vocab_size
    pad_token_id = model.config.pad_token_id

    def check_id(name: str, value: int) -> None:
        if type(value) is not int or not 0 <= value < vocab_size:
            raise ValueError(f"{name} must be a vocabulary ID")

    if start_token_id is not None:
        check_id("start_token_id", start_token_id)
        if start_token_id == pad_token_id or start_token_id == mask_token_id:
            raise ValueError("start_token_id cannot be PAD or MASK")
    if mask_token_id is not None:
        check_id("mask_token_id", mask_token_id)
        if mask_token_id == pad_token_id:
            raise ValueError("MASK and PAD must differ")
    for token_id in sampling.blocked_token_ids:
        check_id("blocked token", token_id)
    for token_id in stop.token_ids:
        check_id("stop token", token_id)
    if prompt_ids.numel() and (torch.any(prompt_ids < 0) or torch.any(prompt_ids >= vocab_size)):
        raise ValueError("prompt_ids contain an out-of-vocabulary ID")

    blocked = set(sampling.blocked_token_ids)
    blocked.add(pad_token_id)
    if mask_token_id is not None:
        blocked.add(mask_token_id)
    if len(blocked) == vocab_size:
        raise ValueError("all output tokens are blocked")
    if stop.mode != "none" and blocked.intersection(stop.token_ids):
        raise ValueError("a stop token cannot be blocked from prediction")
    return replace(sampling, blocked_token_ids=tuple(sorted(blocked)))


def make_result(
    token_ids: torch.Tensor,
    lengths: torch.Tensor,
    finished: torch.Tensor,
    stats: GenerationStats,
) -> GenerationResult:
    """Assemble a generator's result, trimming unused batch-wide tail slots."""
    width = int(lengths.max().item())
    return GenerationResult(
        token_ids=token_ids[:, :width],
        lengths=lengths,
        finish_reason=tuple("eos" if value else "length" for value in finished.tolist()),
        stats=stats,
    )
