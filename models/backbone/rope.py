"""One explicit-position RoPE implementation for all model families."""

from __future__ import annotations

from typing import Callable, Optional, Union, cast

import torch
from torch import nn

from .types import PositionPolicy


def rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    first, second = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class RotaryTables:
    """Tables owned by one core forward, including lazy per-Q/K-dtype copies.

    Each dtype view derives directly from the base tables. TransformerCore
    creates a fresh instance for every forward.
    """

    def __init__(self, base: tuple[torch.Tensor, torch.Tensor]) -> None:
        self.base = base
        self._by_dtype = {base[0].dtype: base}

    def for_dtype(self, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        tables = self._by_dtype.get(dtype)
        if tables is None:
            tables = (self.base[0].to(dtype=dtype), self.base[1].to(dtype=dtype))
            self._by_dtype[dtype] = tables
        return tables


RotaryTableInput = Union[tuple[torch.Tensor, torch.Tensor], RotaryTables]


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        if head_dim <= 0 or head_dim % 2:
            raise ValueError("head_dim must be a positive even integer")
        self.head_dim = head_dim
        self.base = float(base)
        self.register_buffer("inv_freq", self._inverse_frequency(), persistent=False)

    def _inverse_frequency(self) -> torch.Tensor:
        exponent = torch.arange(0, self.head_dim, 2, device="cpu", dtype=torch.float32)
        return cast('torch.Tensor', 1.0 / (self.base ** (exponent / self.head_dim)))

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool=True) -> RotaryEmbedding:
        super()._apply(fn, recurse=recurse)
        if self.inv_freq.dtype is not torch.float32:
            # Rebuild from the base instead of upcasting an already rounded buffer.
            self.inv_freq = self._inverse_frequency().to(device=self.inv_freq.device)
        return self

    def cos_sin(
        self,
        position_ids: torch.Tensor,
        policy: PositionPolicy,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim != 2:
            raise ValueError("position_ids must have shape [batch, sequence]")
        if position_ids.device != self.inv_freq.device:
            raise ValueError("position_ids and RoPE buffer must share a device")
        # Positions, frequencies and tables stay FP32 even with low-precision
        # model weights. Only the returned Q/K tables change dtype.
        with torch.autocast(device_type=position_ids.device.type, enabled=False):
            positions = policy.logical_positions(position_ids)
            frequencies = torch.einsum("bs,d->bsd", positions, self.inv_freq)
            embedding = torch.cat((frequencies, frequencies), dim=-1)
            cosine = embedding.cos()
            sine = embedding.sin()
        return (
            cosine[:, :, None, :].to(dtype=dtype),
            sine[:, :, None, :].to(dtype=dtype),
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        position_ids: torch.Tensor,
        policy: PositionPolicy = PositionPolicy(),
        *,
        rotary_cos_sin: Optional[RotaryTableInput] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query.shape != key.shape:
            raise ValueError("query and key must have identical shapes before RoPE")
        if query.ndim != 4 or query.shape[-1] != self.head_dim:
            raise ValueError("query/key must have shape [batch, sequence, heads, head_dim]")
        if position_ids.shape != query.shape[:2]:
            raise ValueError("position_ids do not match query/key")
        if rotary_cos_sin is None:
            cos, sin = self.cos_sin(position_ids, policy, query.dtype)
        elif isinstance(rotary_cos_sin, RotaryTables):
            # Lazily match the projected Q/K dtype and reuse within this forward.
            cos, sin = rotary_cos_sin.for_dtype(query.dtype)
        else:
            # Standalone callers may supply a plain (cosine, sine) pair.
            cos, sin = rotary_cos_sin
            cos = cos.to(dtype=query.dtype)
            sin = sin.to(dtype=query.dtype)
        return (
            query * cos + rotate_half(query) * sin,
            key * cos + rotate_half(key) * sin,
        )
