"""Autoregressive token generation using explicit cache commits."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, cast

import torch

from kv_cache import CacheMode
from progress import progress_bar

from .config import SamplingConfig, StopConfig
from .runtime import cache_dtype, generation_context, make_result, prepare_request
from .sampling import candidate_probabilities, sample_candidates
from .types import GenerationResult, GenerationStats

if TYPE_CHECKING:
    from models import ARModel


def generate_ar(
    model: ARModel,
    prompt_ids: torch.Tensor,
    *,
    generation_length: int,
    start_token_id: Optional[int] = None,
    sampling: SamplingConfig = SamplingConfig(),
    stop: StopConfig = StopConfig(),
    generator: Optional[torch.Generator] = None,
) -> GenerationResult:
    """Generate G slots; an optional fixed start occupies the first slot."""
    sampling = prepare_request(
        model, prompt_ids, generation_length, start_token_id, sampling, stop
    )
    pad_id = model.config.pad_token_id
    batch_size, prompt_width = prompt_ids.shape
    has_start = start_token_id is not None
    valid_prompt = prompt_ids.ne(pad_id).sum(dim=-1)
    if not has_start and torch.any(valid_prompt == 0):
        raise ValueError("each AR row requires a valid prompt or fixed start token")

    max_attention = model.config.backbone.max_attention_length
    max_position = model.config.backbone.max_position_id
    content_limit = model._content_limit
    if prompt_width + int(has_start) > max_attention:
        raise ValueError("AR prefill width exceeds model attention capacity")
    if torch.any(valid_prompt + generation_length > min(content_limit, max_attention)):
        capacity = "content" if content_limit <= max_attention else "attention"
        raise ValueError(f"AR context exceeds model {capacity} capacity")
    if torch.any(valid_prompt + generation_length - 1 > max_position):
        raise ValueError("AR context exceeds model position capacity")

    device = prompt_ids.device
    result_ids = torch.full(
        (batch_size, generation_length), pad_id, device=device, dtype=torch.long
    )
    lengths = torch.zeros(batch_size, device=device, dtype=torch.long)
    finished = torch.zeros(batch_size, device=device, dtype=torch.bool)
    stats = GenerationStats.empty(batch_size, device)
    if has_start:
        result_ids[:, 0] = start_token_id
        lengths += 1

    prediction_budget = generation_length - int(has_start)
    if prediction_budget == 0:
        return make_result(result_ids, lengths, finished, stats)

    start_column = (
        prompt_ids.new_full((batch_size, 1), start_token_id)
        if has_start else None
    )
    current_ids = (
        torch.cat((prompt_ids, cast(torch.Tensor, start_column)), dim=1) if has_start else prompt_ids
    )
    active_rows = torch.arange(batch_size, device=device)
    stop_ids = torch.tensor(stop.token_ids, device=device, dtype=torch.long)

    with generation_context(model), progress_bar(
            desc="AR generation", total=prediction_budget, unit="token step") as progress:
        session = model.new_cache_session(batch_size, device=device, dtype=cache_dtype(model))
        for position in range(int(has_start), generation_length):
            output = model.generate_step(
                current_ids, session,
                cache_mode=CacheMode.STAGE_APPEND,
                logits_mode="last_valid",
            )
            if output.cache_delta is None or not cast(torch.Tensor, output.logits_valid).all():
                raise RuntimeError("AR prediction requires a valid logit and cache delta per active row")
            session.commit(output.cache_delta)
            stats.record("prediction", active_rows)

            probabilities = candidate_probabilities(output.logits[:, 0], sampling, ar=True)
            chosen = sample_candidates(
                probabilities, greedy=sampling.greedy, generator=generator, ar=True
            )
            result_ids[active_rows, position] = chosen
            lengths[active_rows] += 1
            hit_stop = (
                torch.isin(chosen, stop_ids)
                if stop.mode != "none" else torch.zeros_like(chosen, dtype=torch.bool)
            )
            finished[active_rows] = hit_stop
            if progress is not None:
                progress.update()

            if position + 1 == generation_length:
                break
            keep = (~hit_stop).nonzero(as_tuple=True)[0]
            if keep.numel() == 0:
                break
            if keep.numel() != active_rows.numel():
                session.select_rows(keep)
                active_rows = active_rows.index_select(0, keep)
                chosen = chosen.index_select(0, keep)
            current_ids = chosen[:, None]

    return make_result(result_ids, lengths, finished, stats)
