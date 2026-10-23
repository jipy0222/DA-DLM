"""Autograd bridge for fused vocabulary normalization and target selection."""

from __future__ import annotations

from typing import Sequence, cast

import torch
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from ._autograd_types import GatherContext
from ._config import DEFAULT_LOG_SOFTMAX_GATHER_CONFIG, LogSoftmaxGatherLaunchConfig
from ._extension import load_cuda_extension


class _LogSoftmaxGatherCuda(Function):
    @staticmethod
    def forward(ctx: GatherContext, logits: Tensor, indices: Tensor, blocked: Tensor, require_gradient: bool, threads: int) -> Tensor:
        compute_dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
        # This private buffer holds probabilities, then the one-shot gradient.
        workspace = logits.to(dtype=compute_dtype, copy=True,
                              memory_format=torch.contiguous_format)
        workspace.masked_fill_(blocked[None, None, :], -torch.inf)
        result = load_cuda_extension().logsoftmax_gather(
            workspace, indices, require_gradient, threads)
        if require_gradient:
            ctx.save_for_backward(workspace, indices, blocked)
            ctx.input_dtype = logits.dtype
            ctx.backward_consumed = False
        return result

    @staticmethod
    @once_differentiable  # ty: ignore[dynamic-function-decorator-return]
    def backward(ctx: GatherContext, grad_output: Tensor) -> tuple[Tensor, None, None, None, None]:
        if ctx.backward_consumed:
            raise RuntimeError("CUDA log-softmax gather supports only one backward per forward")
        gradient, indices, blocked = ctx.saved_tensors
        ctx.backward_consumed = True
        if grad_output.numel() == 0:
            gradient.zero_()
        else:
            gradient.mul_(-grad_output.sum(-1, keepdim=True))
            gradient.scatter_add_(-1, indices, grad_output)
            gradient.masked_fill_(blocked[None, None, :], 0)
        return gradient.to(ctx.input_dtype), None, None, None, None


def log_softmax_gather_cuda(
    logits: Tensor,
    select_idx: Tensor,
    *,
    blocked_token_ids: Sequence[int]=(),
    launch_config: LogSoftmaxGatherLaunchConfig = DEFAULT_LOG_SOFTMAX_GATHER_CONFIG,
) -> Tensor:
    """Return selected log probabilities without modifying input logits.

    Inputs are logits [batch, node, vocabulary] and int64 indices
    [batch, node, target]. FP16/BF16 inputs compute and save probabilities in
    FP32; FP64 is preserved. Supports one first-order backward per forward;
    repeated backward and higher-order derivatives are unsupported. Indices
    must be in range.
    """
    if not logits.is_cuda:
        raise ValueError("fused log-softmax gather requires CUDA logits")
    if logits.ndim != 3 or logits.shape[-1] < 1:
        raise ValueError("logits must have shape [batch, node, nonempty vocabulary]")
    if logits.dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
        raise TypeError("logits must use a floating-point score dtype")
    if select_idx.ndim != 3 or select_idx.shape[:2] != logits.shape[:2]:
        raise ValueError("indices must match logits batch and node dimensions")
    if select_idx.dtype != torch.long or select_idx.device != logits.device:
        raise ValueError("indices must be int64 on the logits device")
    if any(token_id < 0 or token_id >= logits.shape[-1] for token_id in blocked_token_ids):
        raise ValueError("blocked_token_ids must be within the vocabulary")
    blocked = torch.zeros(logits.shape[-1], dtype=torch.bool, device=logits.device)
    if blocked_token_ids:
        blocked[list(blocked_token_ids)] = True
    require_gradient = torch.is_grad_enabled() and logits.requires_grad
    return cast('Tensor', _LogSoftmaxGatherCuda.apply(
        logits, select_idx, blocked, require_gradient, launch_config.threads))


__all__ = ["LogSoftmaxGatherLaunchConfig", "log_softmax_gather_cuda"]
