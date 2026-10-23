"""Masked-diffusion corruption and token-weighted reconstruction sums."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from .types import LossTerms


def prepare_mdlm_batch(input_ids: Tensor, *, pad_token_id: int, loss_mask: Tensor | None=None,
                       sample_weight: Tensor | None=None, ignore_bos: bool=True, bos_token_id: int | None=None) -> tuple[Tensor, Tensor]:
    """Return corruption eligibility and target weights, without shifting IDs.

    PAD never participates. Optional ``loss_mask`` supplies nonnegative target
    weights; its zero positions remain clean context. Row ``sample_weight``
    affects accounting only, not corruption. With ``ignore_bos=True``, every
    row must start with the supplied BOS ID; that position stays visible and
    never counts, even if an explicit loss_mask assigns it positive weight.
    Other occurrences of the same ID are ordinary targets (BOS/EOS may alias).
    """
    if input_ids.ndim != 2 or input_ids.shape[1] == 0:
        raise ValueError("input_ids must have shape [batch, nonempty sequence]")
    if ignore_bos:
        if type(bos_token_id) is not int or bos_token_id < 0 or bos_token_id == pad_token_id:
            raise ValueError("ignore_bos=True requires an explicit BOS ID distinct from PAD")
        if not bool(input_ids[:, 0].eq(bos_token_id).all()):
            raise ValueError("ignore_bos=True requires every target row to start with bos_token_id=%d"
                             % bos_token_id)
    valid = input_ids.ne(pad_token_id)
    if loss_mask is None:
        weights = valid.to(torch.float32)
        eligible = valid.clone()
    else:
        if loss_mask.shape != input_ids.shape:
            raise ValueError("loss_mask must match input_ids")
        dtype = torch.float64 if loss_mask.dtype == torch.float64 else torch.float32
        weights = loss_mask.to(dtype).masked_fill(~valid, 0)
        eligible = valid & (weights > 0)
    if ignore_bos:
        eligible[:, 0] = False
        weights[:, 0] = 0
    if sample_weight is not None:
        if sample_weight.shape != input_ids.shape[:1]:
            raise ValueError("sample_weight must have shape [batch]")
        weights = weights * sample_weight[:, None]
    return eligible, weights


def mask_tokens(input_ids: Tensor, mask_probability: Tensor, *, mask_token_id: int,
                corruption_mask: Tensor, uniform: Tensor | None=None, generator: torch.Generator | None=None) -> Tensor:
    """Independently replace eligible tokens by MASK, without modifying inputs.

    ``mask_probability`` is [B, 1] (or broadcastable to [B, L]). An explicit
    [B, L] uniform draw permits deterministic corruption tests and replay.
    """
    if corruption_mask.shape != input_ids.shape:
        raise ValueError("corruption_mask must match input_ids")
    if uniform is None:
        uniform = torch.rand(input_ids.shape, device=input_ids.device, generator=generator)
    elif uniform.shape != input_ids.shape:
        raise ValueError("uniform must match input_ids")
    replace = (uniform <= mask_probability) & corruption_mask
    return torch.where(replace, mask_token_id, input_ids)


def mdlm_objective(logits: Tensor, targets: Tensor, noisy_ids: Tensor,
                   loss_weight: Tensor, loss_mask: Tensor | None=None, *, mask_token_id: int,
                   blocked_token_ids: Sequence[int]=()) -> LossTerms:
    """Sum diffusion-weighted CE at MASK positions; count all target weights.

    Targets and noisy IDs are aligned [B, L]; there is no AR shift. A valid
    corruption leaves each token unchanged or replaces it with MASK. Visible
    tokens have exact carry-over probability one and contribute zero loss.
    ``loss_weight`` is positive and broadcastable to [B, L], usually [B, 1].
    The denominator is NOT the random number of masked tokens.

    Blocked output logits are replaced by -inf, giving them probability zero.
    Active targets must be allowed output classes. Inactive positions are
    zeroed before weighting, so ignored PAD/markers cannot produce inf * 0.
    Computation is FP32, or FP64 for FP64 logits. IDs/weights share the logits
    device. Weights must be finite and nonnegative; blocked IDs are checked at
    configuration and must leave at least one allowed class.
    """
    if logits.ndim != 3 or targets.shape != logits.shape[:2] or noisy_ids.shape != targets.shape:
        raise ValueError("targets and noisy_ids must match [batch, sequence] logits dimensions")
    if loss_mask is not None and loss_mask.shape != targets.shape:
        raise ValueError("loss_mask must match targets")
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=logits.device.type, enabled=False):
        weights = (torch.ones_like(targets, dtype=dtype) if loss_mask is None
                   else loss_mask.to(dtype))
        count = weights.sum()
        active = noisy_ids.eq(mask_token_id) & weights.ne(0)
        scores = logits.to(dtype)
        if blocked_token_ids:
            blocked = torch.zeros(scores.shape[-1], device=scores.device, dtype=torch.bool)
            blocked[list(blocked_token_ids)] = True
            scores = scores.masked_fill(blocked, -torch.inf)
        nll = -F.log_softmax(scores, dim=-1).gather(-1, targets[..., None]).squeeze(-1)
        nll = nll.masked_fill(~active, 0)
        return LossTerms((nll * loss_weight.to(dtype) * weights).sum(), count)
