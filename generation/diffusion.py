"""One-block denoising with per-sample prediction reuse."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, cast

import torch

from progress import track

from .config import DiffusionConfig, SamplingConfig, StopConfig
from .reveal import reveal_tokens
from .sampling import candidate_probabilities, sample_candidates
from .stopping import apply_block_stop
from .types import GenerationStats


class GenerationIncompleteError(RuntimeError):
    """The update budget ended with required target positions still masked."""

    def __init__(self, row_ids: torch.Tensor, remaining: torch.Tensor, num_steps: int) -> None:
        self.row_ids = row_ids.detach().clone()
        self.remaining = remaining.detach().clone()
        super().__init__(
            f"generation incomplete after {num_steps} updates: "
            f"rows={row_ids.tolist()}, remaining_mask={remaining.tolist()}"
        )


@dataclass
class BlockResult:
    token_ids: torch.Tensor
    lengths: torch.Tensor
    finished: torch.Tensor


def denoise_block(current_ids: torch.Tensor, predict: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], *, mask_token_id: int, pad_token_id: int,
                  sampling: SamplingConfig, diffusion: DiffusionConfig,
                  stop: StopConfig, stats: GenerationStats, row_ids: torch.Tensor, generator: torch.Generator | None=None) -> BlockResult:
    """Predict only unfinished rows whose current input has no reusable prediction.

    ``predict(ids, rows)`` receives row positions within this block's batch.
    Its model, conditioning and history must stay fixed for this call. The
    caller commits completed content only after this function returns.
    """
    current = current_ids.clone()
    batch, width = current.shape
    eligible = current.eq(mask_token_id)
    prediction_valid = torch.zeros(batch, device=current.device, dtype=torch.bool)
    boundary = torch.full((batch,), width, device=current.device, dtype=torch.long)
    probabilities = None

    for step in track(range(diffusion.num_steps), desc="Denoising", unit="update"):
        unfinished = current.eq(mask_token_id).any(dim=-1)
        active = unfinished.nonzero(as_tuple=True)[0]
        if active.numel() == 0:
            break
        needed = (unfinished & ~prediction_valid).nonzero(as_tuple=True)[0]
        if needed.numel():
            if needed.numel() == batch:
                probabilities = None
                selected = current
            else:
                selected = current[needed]
            fresh = candidate_probabilities(predict(selected, needed), sampling)
            stats.record("prediction", row_ids[needed])
            if probabilities is None:
                # Initially every unfinished row needs a prediction. The empty
                # slots, if any, belong only to already completed rows.
                if needed.numel() == batch:
                    probabilities = fresh
                else:
                    probabilities = fresh.new_empty(batch, width, fresh.shape[-1])
                    probabilities.index_copy_(0, needed, fresh)
            else:
                probabilities.index_copy_(0, needed, fresh)
            prediction_valid[needed] = True
            del fresh

        previous = current[active]
        active_probabilities = probabilities if active.numel() == batch else cast(torch.Tensor, probabilities)[active]
        candidates = sample_candidates(cast(torch.Tensor, active_probabilities), greedy=sampling.greedy,
                                       generator=generator)
        updated = reveal_tokens(previous, candidates, cast(torch.Tensor, active_probabilities),
                                mask_token_id=mask_token_id, step=step,
                                config=diffusion, generator=generator)
        del active_probabilities
        updated, accepted_stop = apply_block_stop(
            updated, eligible[active], stop, pad_token_id)
        prediction_valid[active] = updated.eq(previous).all(dim=-1)
        current.index_copy_(0, active, updated)
        boundary[active] = accepted_stop
        stats.update_steps[row_ids[active]] += 1

    remaining = current.eq(mask_token_id).sum(dim=-1)
    failed = remaining.ne(0)
    if failed.any():
        raise GenerationIncompleteError(row_ids[failed], remaining[failed], diffusion.num_steps)
    return BlockResult(current, (boundary + 1).clamp_max(width), boundary.lt(width))
