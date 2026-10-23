"""Vocabulary-normalized target scores with a PyTorch reference backend."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from graph.runtime import GraphBackend, validate_backend


def log_softmax_gather(logits: Tensor, target_ids: Tensor, *,
                       blocked_token_ids: Sequence[int]=(), backend: GraphBackend="auto") -> Tensor:
    """Map [blocks, nodes, vocabulary] to [blocks, targets, nodes].

    CUDA supports one first-order backward per forward. Explicit PyTorch mode
    retains ordinary autograd, including repeated and higher-order derivatives.
    Forbidden vocabulary classes are removed before normalization.
    """
    validate_backend(backend)
    if logits.ndim != 3 or logits.shape[-1] < 1:
        raise ValueError("logits must have shape [blocks, nodes, nonempty vocabulary]")
    if target_ids.ndim != 2 or target_ids.shape[0] != logits.shape[0]:
        raise ValueError("target_ids must match [blocks, targets]")
    if target_ids.dtype != torch.long or target_ids.device != logits.device:
        raise ValueError("target_ids must be int64 on the logits device")
    if logits.dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
        raise TypeError("logits must use a floating-point score dtype")
    if any(token_id < 0 or token_id >= logits.shape[-1] for token_id in blocked_token_ids):
        raise ValueError("blocked_token_ids must be within the vocabulary")
    indices = target_ids[:, None, :].expand(-1, logits.shape[1], -1)
    if backend == "cuda" or (backend == "auto" and logits.is_cuda):
        from graph.backends.cuda._log_softmax_gather import log_softmax_gather_cuda

        return log_softmax_gather_cuda(
            logits, indices, blocked_token_ids=blocked_token_ids).transpose(1, 2)
    compute_dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=logits.device.type, enabled=False):
        scores = logits.to(compute_dtype)
        if blocked_token_ids:
            blocked = torch.zeros(logits.shape[-1], dtype=torch.bool, device=logits.device)
            blocked[list(blocked_token_ids)] = True
            scores = scores.masked_fill(blocked, -torch.inf)
        return F.log_softmax(scores, dim=-1).gather(-1, indices).transpose(1, 2)
