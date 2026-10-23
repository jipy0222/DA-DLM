"""Plain PyTorch autoregressive preparation and negative-log-likelihood sums.

Callers supply token weights and blocked vocabulary IDs, then normalize the sums.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F

from .types import LossTerms


def prepare_ar_batch(
    input_ids: Tensor,
    loss_mask: Optional[Tensor] = None,
    sample_weight: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Shift an already formatted clean sequence for next-token prediction.

    Args:
        input_ids: Long ``[B, L]`` token IDs, with ``L >= 2``.
        loss_mask: Optional nonnegative ``[B, L]`` target-position weights.
            Source/BOS/PAD exclusion is the caller's task-formatting decision.
        sample_weight: Optional nonnegative ``[B]`` row weights. A zero weight
            excludes padding rows from evaluation sums without changing shapes.

    Returns ``(input_ids[:, :-1], input_ids[:, 1:], shifted_weights)``. The two
    token tensors are borrowed views; this function never mutates its inputs.
    Token exclusion is controlled entirely by the supplied weights. Tensors
    must share a device.
    """
    if input_ids.ndim != 2 or input_ids.shape[1] < 2:
        raise ValueError("input_ids must have shape [batch, sequence >= 2]")
    if loss_mask is None:
        loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
    elif loss_mask.shape != input_ids.shape:
        raise ValueError("loss_mask must match input_ids")
    else:
        loss_mask = loss_mask.to(torch.float64 if loss_mask.dtype == torch.float64
                                 else torch.float32)
    if sample_weight is not None:
        if sample_weight.shape != input_ids.shape[:1]:
            raise ValueError("sample_weight must have shape [batch]")
        sample_weight = sample_weight.to(
            torch.float64 if sample_weight.dtype == torch.float64 else torch.float32)
        loss_mask = loss_mask * sample_weight[:, None]
    return input_ids[:, :-1], input_ids[:, 1:], loss_mask[:, 1:]


def ar_objective(
    logits: Tensor,
    targets: Tensor,
    loss_mask: Optional[Tensor] = None,
    *,
    blocked_token_ids: Tuple[int, ...] = (),
) -> LossTerms:
    """Return weighted AR NLL and token-count sums, without dividing them.

    ``logits`` are raw ``[B, S, V]`` vocabulary scores, and ``targets`` are
    aligned long ``[B, S]`` IDs. Optional ``loss_mask`` supplies aligned finite,
    nonnegative weights; omitted weights are all one. Targets, weights and
    logits must share a device, and targets must be inside the vocabulary.

    Blocked logits are replaced by ``-inf`` before log-softmax, giving those
    classes probability zero. Scored targets must be allowed output classes.
    Ignored targets are zeroed before weighting to avoid ``inf * 0``. The token
    schema blocks MASK/PAD and task-internal markers, not ordinary BOS/EOS.

    ``blocked_token_ids`` is a tuple of unique, in-range Python integers and
    must leave at least one class unblocked. Validate these IDs when configuring
    training.

    Computation uses FP32, or FP64 for FP64 logits. Inputs are never modified.
    """
    if logits.ndim != 3 or targets.shape != logits.shape[:2]:
        raise ValueError("targets must match [batch, sequence] logits dimensions")
    if loss_mask is not None and loss_mask.shape != targets.shape:
        raise ValueError("loss_mask must match targets")

    compute_dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=logits.device.type, enabled=False):
        weights = (torch.ones(targets.shape, device=logits.device, dtype=compute_dtype)
                   if loss_mask is None else loss_mask.to(dtype=compute_dtype))
        token_count = weights.sum()
        if not torch.any(weights):
            return LossTerms(logits[..., :0].to(dtype=compute_dtype).sum(), token_count)
        scores = logits.to(dtype=compute_dtype)
        if blocked_token_ids:
            blocked = torch.zeros(scores.shape[-1], device=scores.device, dtype=torch.bool)
            blocked[list(blocked_token_ids)] = True
            scores = scores.masked_fill(blocked, -torch.inf)
        log_probs = F.log_softmax(scores, dim=-1)
        nll = -log_probs.gather(-1, targets[..., None]).squeeze(-1)
        nll = nll.masked_fill(weights == 0, 0.0)
        return LossTerms((nll * weights).sum(), token_count)
