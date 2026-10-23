"""Multi-head self-attention with explicit layout and RoPE inputs."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Callable, Optional, cast

import torch
from torch import nn
from torch.nn import functional as F

from kv_cache.packing import CachePackingPlan, make_cache_packing_plan

from .config import AttentionBackend, TransformerConfig
from .rope import RotaryEmbedding, RotaryTableInput
from .types import AttentionLayout, PositionPolicy

if TYPE_CHECKING:
    from torch.nn.attention.flex_attention import BlockMask


@dataclass(frozen=True)
class AttentionResult:
    hidden_states: torch.Tensor
    current_key: torch.Tensor
    current_value: torch.Tensor


def build_flex_block_mask(layout: AttentionLayout) -> BlockMask:
    """Materialize one backend object that every layer in a core call can share."""

    try:
        from torch.nn.attention.flex_attention import create_block_mask
    except ImportError as error:
        raise RuntimeError(
            "FlexAttention is unavailable; explicitly select the SDPA backend"
        ) from error

    if layout.structural_allowed is not None:
        structural = layout.structural_allowed
        key_valid = layout.key_valid

        def mask_mod(batch_index: torch.Tensor, head_index: torch.Tensor, query_index: torch.Tensor, key_index: torch.Tensor) -> torch.Tensor:
            del head_index
            return structural[query_index, key_index] & key_valid[batch_index, key_index]
    else:
        allowed = layout.allowed

        def mask_mod(batch_index: torch.Tensor, head_index: torch.Tensor, query_index: torch.Tensor, key_index: torch.Tensor) -> torch.Tensor:
            del head_index
            return allowed[batch_index, query_index, key_index]

    operation = compiled_cuda_block_mask() if layout.key_valid.is_cuda else create_block_mask
    return operation(
        mask_mod,
        B=layout.batch_size,
        H=None,
        Q_LEN=layout.query_length,
        KV_LEN=layout.key_length,
        device=str(layout.key_valid.device),
    )


@lru_cache(maxsize=1)
def compiled_cuda_block_mask() -> Callable[..., BlockMask]:
    """Fuse CUDA relation construction and reductions separately from attention.

    The cache holds only the compiled callable; relation tensors remain local
    to each request.
    """

    from torch.nn.attention.flex_attention import create_block_mask

    return torch.compile(create_block_mask, fullgraph=True, dynamic=False)


@lru_cache(maxsize=1)
def compiled_cuda_flex_attention() -> Callable[..., torch.Tensor]:
    """Return the tuned CUDA FlexAttention entry point.

    ``max-autotune-no-cudagraphs`` makes Inductor benchmark candidate Triton
    schedules when a new compiled signature is first encountered and cache the
    selected schedule.  It is therefore important that callers warm up every
    production shape before measuring steady-state performance.
    """

    try:
        from torch.nn.attention.flex_attention import flex_attention
    except ImportError as error:
        raise RuntimeError(
            "FlexAttention is unavailable; explicitly select the SDPA backend"
        ) from error
    return cast(Callable[..., torch.Tensor], torch.compile(
        flex_attention,
        fullgraph=True,
        mode="max-autotune-no-cudagraphs",
        dynamic=False,
    ))


class MultiheadSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_heads = config.n_heads
        self.head_dim = config.hidden_size // config.n_heads
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.out = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def _project(
        self,
        hidden_states: torch.Tensor,
        rotary: RotaryEmbedding,
        position_ids: torch.Tensor,
        position_policy: PositionPolicy,
        rotary_cos_sin: Optional[RotaryTableInput] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, sequence_length, _ = hidden_states.shape
        qkv = self.qkv(hidden_states).view(
            batch, sequence_length, 3, self.n_heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query, key = rotary(
            query, key, position_ids, position_policy,
            rotary_cos_sin=rotary_cos_sin,
        )
        # Give V independent storage because autograd and staged cache deltas
        # may retain it after Q/K have been replaced. clone() also handles
        # singleton dimensions whose slices are already contiguous.
        value = value.clone(memory_format=torch.contiguous_format)
        return cast('tuple[torch.Tensor, torch.Tensor, torch.Tensor]', (query, key, value))

    def _sdpa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        attended = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
            scale=self.head_dim ** -0.5,
        )
        return attended.transpose(1, 2).contiguous()

    def _flex(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layout: AttentionLayout,
        block_mask: Optional[BlockMask],
    ) -> torch.Tensor:
        try:
            from torch.nn.attention.flex_attention import flex_attention
        except ImportError as error:
            raise RuntimeError(
                "FlexAttention is unavailable; explicitly select the SDPA backend"
            ) from error
        if query.is_cuda and self.head_dim < 16:
            raise RuntimeError(
                "compiled CUDA FlexAttention requires attention head_dim >= 16; "
                "select SDPA explicitly for smaller test or research models"
            )
        if block_mask is None:
            block_mask = build_flex_block_mask(layout)
        operation = compiled_cuda_flex_attention() if query.is_cuda else flex_attention
        return cast(torch.Tensor, operation(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            block_mask=block_mask,
            scale=self.head_dim ** -0.5,
        )).transpose(1, 2).contiguous()

    def _merge_cached_kv(
        self,
        past_key: torch.Tensor,
        past_value: torch.Tensor,
        past_valid: torch.Tensor,
        current_key: torch.Tensor,
        current_value: torch.Tensor,
        current_valid: torch.Tensor,
        packing_plan: Optional[CachePackingPlan] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if packing_plan is None:
            # Independent calls may have holes in both history and current K/V.
            valid = torch.cat((past_valid, current_valid), dim=1)
            packing_plan = make_cache_packing_plan(
                past_valid.new_zeros(past_valid.shape[0], dtype=torch.long), valid
            )
            current_key = torch.cat((past_key.transpose(1, 2), current_key), dim=1)
            current_value = torch.cat((past_value.transpose(1, 2), current_value), dim=1)
            past_key = past_key[:, :, :0]
            past_value = past_value[:, :, :0]
        packed_key = packing_plan.merge(past_key, current_key.transpose(1, 2))
        packed_value = packing_plan.merge(past_value, current_value.transpose(1, 2))
        return packed_key.transpose(1, 2), packed_value.transpose(1, 2), packing_plan.valid

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        rotary: RotaryEmbedding,
        position_ids: torch.Tensor,
        position_policy: PositionPolicy,
        layout: AttentionLayout,
        backend: AttentionBackend,
        past_key: Optional[torch.Tensor] = None,
        past_value: Optional[torch.Tensor] = None,
        past_valid: Optional[torch.Tensor] = None,
        flex_block_mask: Optional[BlockMask] = None,
        rotary_cos_sin: Optional[RotaryTableInput] = None,
        packing_plan: Optional[CachePackingPlan] = None,
        sdpa_mask: Optional[torch.Tensor] = None,
    ) -> AttentionResult:
        if layout.batch_size != hidden_states.shape[0]:
            raise ValueError("attention layout batch size does not match input")
        if layout.query_length != hidden_states.shape[1]:
            raise ValueError("attention layout query length does not match input")
        query, current_key, current_value = self._project(
            hidden_states, rotary, position_ids, position_policy, rotary_cos_sin
        )
        supplied = (past_key is not None, past_value is not None, past_valid is not None)
        if any(supplied) and not all(supplied):
            raise ValueError("past_key, past_value, and past_valid must be supplied together")
        if past_key is not None:
            past_value = cast(torch.Tensor, past_value)
            past_valid = cast(torch.Tensor, past_valid)
            expected = (
                hidden_states.shape[0],
                self.n_heads,
                past_key.shape[2],
                self.head_dim,
            )
            if past_key.shape != expected or past_value.shape != expected:
                raise ValueError("cached K/V have an incompatible shape")
            if past_valid.shape != (hidden_states.shape[0], past_key.shape[2]):
                raise ValueError("past_valid has an incompatible shape")
            if (
                past_key.device != current_key.device
                or past_key.dtype != current_key.dtype
                or past_value.device != current_value.device
                or past_value.dtype != current_value.dtype
            ):
                raise ValueError("cached and current K/V must share device and dtype")
            key, value, packed_valid = self._merge_cached_kv(
                past_key,
                past_value,
                past_valid,
                current_key,
                current_value,
                layout.query_valid,
                packing_plan,
            )
            # The core checks its shared plan once, before entering the layers.
            if packing_plan is None and not torch.equal(layout.key_valid, packed_valid):
                raise ValueError("attention layout does not match packed cached K/V")
        else:
            key = current_key
            value = current_value

        if layout.key_length != key.shape[1]:
            raise ValueError("attention layout key length does not match K/V")

        if backend is AttentionBackend.SDPA:
            if flex_block_mask is not None:
                raise ValueError("an SDPA call cannot receive a Flex BlockMask")
            if sdpa_mask is None:
                # Standalone attention/block calls may omit core preparation.
                sdpa_mask = layout.as_sdpa_mask()
            attended = self._sdpa(query, key, value, sdpa_mask)
        elif backend is AttentionBackend.FLEX:
            if past_key is not None:
                raise ValueError("generation with FlexAttention is not supported")
            attended = self._flex(query, key, value, layout, flex_block_mask)
        else:
            raise ValueError("unsupported attention backend: %s" % backend)
        attended = attended.view(
            hidden_states.shape[0], hidden_states.shape[1], self.hidden_size
        )
        return AttentionResult(self.out(attended), current_key, current_value)
