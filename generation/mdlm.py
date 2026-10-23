"""Full-sequence masked diffusion generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from models.backbone import ConditioningMode

from .config import DiffusionConfig, SamplingConfig, StopConfig
from .diffusion import denoise_block
from .runtime import generation_context, make_result, prepare_request
from .types import GenerationResult, GenerationStats

if TYPE_CHECKING:
    from models import MDLMModel


def generate_mdlm(model: MDLMModel, prompt_ids: torch.Tensor, *, generation_length: int,
                  mask_token_id: int, diffusion: DiffusionConfig,
                  start_token_id: int | None=None, sampling: SamplingConfig=SamplingConfig(),
                  stop: StopConfig=StopConfig(), generator: torch.Generator | None=None) -> GenerationResult:
    """Generate one target block while keeping external prompt tokens fixed.

    The target budget includes an optional fixed start token. Only rows needing
    new predictions run the model; each such row recomputes its whole prefix
    and target with bidirectional attention.
    """
    sampling = prepare_request(model, prompt_ids, generation_length, start_token_id,
                               sampling, stop, mask_token_id=mask_token_id)
    if model.config.conditioning is not ConditioningMode.ZERO_TIME_COMPAT:
        raise ValueError("diffusion generation requires ZERO_TIME_COMPAT conditioning")
    if prompt_ids.shape[1] + generation_length > model.config.backbone.max_attention_length:
        raise ValueError("prompt and target slots exceed max_attention_length")
    valid_prompt = prompt_ids.ne(model.config.pad_token_id).sum(dim=-1)
    if torch.any(valid_prompt + generation_length > model._content_limit):
        raise ValueError("prompt and generation budget exceed content capacity")
    if torch.any(valid_prompt + generation_length - 1 > model.config.backbone.max_position_id):
        raise ValueError("prompt and generation budget exceed max_position_id")
    batch = prompt_ids.shape[0]
    stats = GenerationStats.empty(batch, prompt_ids.device)
    current = prompt_ids.new_full((batch, generation_length), mask_token_id)
    if start_token_id is not None:
        current[:, 0] = start_token_id
        if generation_length == 1:
            lengths = prompt_ids.new_ones(batch)
            finished = torch.zeros(batch, device=prompt_ids.device, dtype=torch.bool)
            return make_result(current, lengths, finished, stats)
    rows = torch.arange(batch, device=prompt_ids.device)

    def predict(ids: torch.Tensor, selected_rows: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat((prompt_ids[selected_rows], ids), dim=1)
        return cast(torch.Tensor, model(inputs).logits)[:, -generation_length:]

    with generation_context(model):
        block = denoise_block(current, predict, mask_token_id=mask_token_id,
                              pad_token_id=model.config.pad_token_id,
                              sampling=sampling, diffusion=diffusion, stop=stop,
                              stats=stats, row_ids=rows, generator=generator)
    slots = torch.arange(generation_length, device=prompt_ids.device)
    output = block.token_ids.masked_fill(slots[None] >= block.lengths[:, None],
                                         model.config.pad_token_id)
    return make_result(output, block.lengths, block.finished, stats)
