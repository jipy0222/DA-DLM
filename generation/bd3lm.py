"""Blockwise diffusion with selected-row predictions and explicit K/V commits."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from kv_cache import CacheDelta, CacheMode, CacheSession
from models.backbone import ConditioningMode
from progress import progress_bar

from .config import DiffusionConfig, SamplingConfig, StopConfig
from .diffusion import denoise_block
from .runtime import cache_dtype, generation_context, make_result, prepare_request
from .types import GenerationResult, GenerationStats

if TYPE_CHECKING:
    from models import BD3LMModel


def generate_bd3lm(model: BD3LMModel, prompt_ids: torch.Tensor, *, generation_length: int,
                   mask_token_id: int, diffusion: DiffusionConfig,
                   start_token_id: int | None=None, sampling: SamplingConfig=SamplingConfig(),
                   stop: StopConfig=StopConfig(), generator: torch.Generator | None=None) -> GenerationResult:
    """Generate the target budget, including an optional start token.

    Full blocks use the model's block size; the final block uses the remaining
    target slots without padding. Each block has the same update-step budget.
    Read-only prediction sessions may select a subset of the main cache rows.
    Completed blocks are recomputed and committed against the main session
    only for rows that need a subsequent block. Final blocks and stopped rows
    do not commit; temporary prediction subset sessions never commit into it.
    """
    block_size = model.config.block_size
    sampling = prepare_request(model, prompt_ids, generation_length, start_token_id,
                               sampling, stop, mask_token_id=mask_token_id)
    if model.config.conditioning is not ConditioningMode.ZERO_TIME_COMPAT:
        raise ValueError("diffusion generation requires ZERO_TIME_COMPAT conditioning")
    pad = model.config.pad_token_id
    valid_prompt = prompt_ids.ne(pad).sum(dim=-1)
    if prompt_ids.shape[1] > model.config.backbone.max_attention_length:
        raise ValueError("prompt slots exceed max_attention_length")
    content_limit = model._content_limit
    attention_limit = model.config.backbone.max_attention_length
    if torch.any(valid_prompt + generation_length > min(content_limit, attention_limit)):
        capacity = "content" if content_limit <= attention_limit else "attention"
        raise ValueError(f"prompt and generation budget exceed {capacity} capacity")
    if torch.any(valid_prompt + generation_length - 1 > model.config.backbone.max_position_id):
        raise ValueError("prompt and generation budget exceed max_position_id")
    batch = prompt_ids.shape[0]
    stats = GenerationStats.empty(batch, prompt_ids.device)
    output = prompt_ids.new_full((batch, generation_length), pad)
    lengths = prompt_ids.new_full((batch,), generation_length)
    finished = torch.zeros(batch, device=prompt_ids.device, dtype=torch.bool)
    if start_token_id is not None:
        output[:, 0] = start_token_id
        if generation_length == 1:
            return make_result(output, lengths, finished, stats)

    with generation_context(model), progress_bar(
            desc="BD3LM generation", total=(generation_length + block_size - 1) // block_size,
            unit="block") as progress:
        original_rows = torch.arange(batch, device=prompt_ids.device)
        session = model.new_cache_session(batch, dtype=cache_dtype(model))
        if torch.any(valid_prompt):
            prefill = model.prefill_prompt(prompt_ids, session)
            session.commit(cast(CacheDelta, prefill.cache_delta))
            del prefill
            stats.record("prefill", original_rows)

        for offset in range(0, generation_length, block_size):
            width = min(block_size, generation_length - offset)
            current = prompt_ids.new_full((original_rows.numel(), width), mask_token_id)
            if offset == 0 and start_token_id is not None:
                current[:, 0] = start_token_id

            def predict(ids: torch.Tensor, selected_rows: torch.Tensor) -> torch.Tensor:
                if selected_rows.numel() == session.cache.batch_size:
                    selected_session = session
                else:
                    selected_session = CacheSession(session.cache.select_rows(selected_rows))
                return model.generation_step(
                    ids, selected_session, cache_mode=CacheMode.READ_ONLY).logits

            block = denoise_block(current, predict, mask_token_id=mask_token_id,
                                  pad_token_id=pad, sampling=sampling,
                                  diffusion=diffusion, stop=stop, stats=stats,
                                  row_ids=original_rows, generator=generator)
            output[original_rows, offset:offset + width] = block.token_ids
            lengths[original_rows] = offset + block.lengths
            finished[original_rows] = block.finished
            if progress is not None:
                progress.update()

            keep = (~block.finished).nonzero(as_tuple=True)[0]
            if offset + width == generation_length or keep.numel() == 0:
                break
            committed_ids = block.token_ids
            if keep.numel() != original_rows.numel():
                session.select_rows(keep)
                original_rows = original_rows[keep]
                committed_ids = block.token_ids[keep]
            committed = model.encode_generation(
                committed_ids, session, cache_mode=CacheMode.STAGE_APPEND)
            session.commit(cast(CacheDelta, committed.cache_delta))
            del committed
            stats.record("commit", original_rows)

    slots = torch.arange(generation_length, device=prompt_ids.device)
    output.masked_fill_(slots[None] >= lengths[:, None], pad)
    return make_result(output, lengths, finished, stats)
