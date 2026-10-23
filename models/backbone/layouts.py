"""Backend-independent attention relations; True means a key is visible.

Full-sequence builders preserve the input order on both axes. Cached builders
preserve query slots but pack valid keys in input order, matching cached K/V.
Padding keys are excluded; padding query outputs have no semantic meaning.
Position IDs, token identities, and cache mutations belong to callers.
"""

from __future__ import annotations

from typing import cast

import torch

from kv_cache import CacheRequest, KVCache

from .config import AttentionKind
from .types import AttentionLayout


def positions_from_valid(valid: torch.Tensor, *, scale: int = 1) -> torch.Tensor:
    """Default integer positions: padding does not advance the position count."""
    if valid.dtype is not torch.bool or valid.ndim != 2:
        raise TypeError("valid must be a boolean [batch, sequence] tensor")
    if scale <= 0:
        raise ValueError("scale must be positive")
    positions = valid.long().cumsum(dim=-1) - 1
    return positions.clamp_min(0) * scale


def _validate_valid(*tensors: torch.Tensor) -> None:
    if any(tensor.dtype is not torch.bool for tensor in tensors):
        raise TypeError("validity tensors must be boolean")
    if any(tensor.ndim != 2 for tensor in tensors):
        raise ValueError("validity tensors must have shape [batch, sequence]")
    reference = tensors[0]
    if any(tensor.shape[0] != reference.shape[0] for tensor in tensors[1:]):
        raise ValueError("validity batch sizes must match")
    if any(tensor.device != reference.device for tensor in tensors[1:]):
        raise ValueError("validity tensors must share a device")


def causal_layout(valid: torch.Tensor) -> AttentionLayout:
    """AR full sequence: a query sees valid keys at or before its input slot."""
    _validate_valid(valid)
    sequence_length = valid.shape[1]
    causal = torch.ones(
        sequence_length,
        sequence_length,
        dtype=torch.bool,
        device=valid.device,
    ).tril()
    allowed = causal[None] & valid[:, None, :]
    return AttentionLayout(allowed, valid, valid, AttentionKind.CAUSAL)


def bidirectional_layout(valid: torch.Tensor) -> AttentionLayout:
    """MDLM full sequence: every query sees every valid key (a broadcast view)."""
    _validate_valid(valid)
    allowed = valid[:, None, :].expand(-1, valid.shape[1], -1)
    return AttentionLayout(
        allowed, valid, valid, AttentionKind.BIDIRECTIONAL
    )


def block_structured_layout(
    *,
    current_valid: torch.Tensor,
    context_valid: torch.Tensor,
    current_block_ids: torch.Tensor,
    context_block_ids: torch.Tensor,
) -> AttentionLayout:
    """BD3LM/DADLM full sequence over ``[current | context]``.

    Current queries see their own current block and strictly earlier context
    blocks. Context queries see same/earlier context blocks and no current keys.
    Block IDs are shared [length] tensors, independent of RoPE position IDs.
    Keep one shared [Q,K] structural table. Flex combines
    it with key validity in its callback; only SDPA/inspection needs [B,Q,K].
    """

    _validate_valid(current_valid, context_valid)
    current_len = current_valid.shape[1]
    context_len = context_valid.shape[1]
    if current_block_ids.shape != (current_len,):
        raise ValueError("current_block_ids must have shape [current_length]")
    if context_block_ids.shape != (context_len,):
        raise ValueError("context_block_ids must have shape [context_length]")
    if current_block_ids.dtype is not torch.long:
        raise TypeError("current_block_ids must use torch.long")
    if context_block_ids.dtype is not torch.long:
        raise TypeError("context_block_ids must use torch.long")
    if torch.any(current_block_ids < 0) or torch.any(context_block_ids < 0):
        raise ValueError("block IDs must be non-negative")
    if current_block_ids.device != current_valid.device:
        raise ValueError("current block IDs must share the layout device")
    if context_block_ids.device != current_valid.device:
        raise ValueError("context block IDs must share the layout device")

    all_valid = torch.cat((current_valid, context_valid), dim=1)
    is_context = torch.cat(
        (
            torch.zeros(current_len, dtype=torch.bool, device=all_valid.device),
            torch.ones(context_len, dtype=torch.bool, device=all_valid.device),
        )
    )
    block_ids = torch.cat((current_block_ids, context_block_ids))

    query_context = is_context[:, None]
    key_context = is_context[None, :]
    query_block = block_ids[:, None]
    key_block = block_ids[None, :]

    block_diagonal = (query_block == key_block) & (
        query_context == key_context
    )
    offset_block_causal = (
        (query_block > key_block) & ~query_context & key_context
    )
    block_causal = (
        (query_block >= key_block) & query_context & key_context
    )
    structural = block_diagonal | offset_block_causal | block_causal
    return AttentionLayout(
        None,
        all_valid,
        all_valid,
        AttentionKind.BLOCK_STRUCTURED_SELF_ATTENTION,
        structural_allowed=structural,
    )


def cached_generation_layout(
    *,
    request: CacheRequest,
    causal_current: bool,
    kind: AttentionKind,
) -> AttentionLayout:
    """Build a cached relation using this request's prepared query mask and plan."""
    current_valid, plan = request.prepared_packing()
    return _cached_relation(
        cast(KVCache, request.cache).lengths, current_valid, plan.valid,
        causal_current=causal_current, kind=kind,
    )


def cached_generation_layout_from_valid(
    *,
    cache_valid: torch.Tensor,
    current_valid: torch.Tensor,
    causal_current: bool,
    kind: AttentionKind,
) -> AttentionLayout:
    """Build a packed-history relation from independent validity masks.

    History is counted and packed; holes in cache_valid do not retain slots.
    """
    _validate_valid(current_valid, cache_valid)
    cache_lengths = cache_valid.sum(dim=-1)
    total_lengths = cache_lengths + current_valid.sum(dim=-1)
    key_length = int(total_lengths.max().item())
    key_valid = torch.arange(key_length, device=current_valid.device)[None] < total_lengths[:, None]
    return _cached_relation(
        cache_lengths, current_valid, key_valid,
        causal_current=causal_current, kind=kind,
    )


def _cached_relation(
    cache_lengths: torch.Tensor,
    current_valid: torch.Tensor,
    key_valid: torch.Tensor,
    *,
    causal_current: bool,
    kind: AttentionKind,
) -> AttentionLayout:
    """Current queries over packed committed history followed by current keys.

    Every valid query sees all committed history. With causal_current=True,
    current keys are token-causal (AR). Otherwise, ALL current valid tokens are
    one bidirectional generation block (BD/DAD preview or final content commit).
    The latter must not bundle separate generation blocks in one call.

    History may be empty: a fixed BOS can belong to the first current block.
    This builder neither infers block boundaries nor decides whether K/V commit.
    """

    batch, query_length = current_valid.shape
    key_length = key_valid.shape[1]
    if causal_current:
        key_slot = torch.arange(key_length, device=current_valid.device)[None, None]
        history_end = cache_lengths[:, None, None]
        query_rank = current_valid.long().cumsum(dim=-1) - 1
        history = key_slot < history_end
        current = (
            (key_slot >= history_end)
            & (key_slot <= history_end + query_rank[:, :, None])
            & current_valid[:, :, None]
        )
        allowed = (history | current) & key_valid[:, None, :]
    else:
        # All current queries see every valid key. A broadcast view suffices.
        allowed = key_valid[:, None, :].expand(batch, query_length, key_length)
    return AttentionLayout(allowed, current_valid, key_valid, kind)


def block_causal_prefill_layout(
    *,
    valid: torch.Tensor,
    block_ids: torch.Tensor,
) -> AttentionLayout:
    """BD/DAD prompt prefill into an empty cache, with explicit prompt blocks.

    block_ids is [sequence] (shared) or [batch, sequence] (per sample).
    A prompt query sees valid prompt keys with key_block_id <= query_block_id.
    Query slots stay intact; key block IDs are gathered with the SAME validity
    and order as cached K/V. Packing never changes a token's block membership.
    """

    _validate_valid(valid)
    batch, query_length = valid.shape
    if block_ids.dtype is not torch.long or block_ids.device != valid.device:
        raise TypeError("block_ids must be torch.long on the validity device")
    if block_ids.shape == (query_length,):
        query_blocks = block_ids[None].expand(batch, -1)
    elif block_ids.shape == valid.shape:
        query_blocks = block_ids
    else:
        raise ValueError("block_ids must have shape [sequence] or [batch, sequence]")

    lengths = valid.sum(dim=-1)
    key_length = int(lengths.max().item())
    key_valid = torch.arange(key_length, device=valid.device)[None] < lengths[:, None]
    # Stable ordering packs valid keys while preserving each token's original
    # block membership.
    key_slots = (~valid).to(torch.int8).argsort(dim=-1, stable=True)[:, :key_length]
    key_blocks = query_blocks.gather(1, key_slots)
    allowed = (key_blocks[:, None, :] <= query_blocks[:, :, None]) & key_valid[:, None, :]
    return AttentionLayout(
        allowed,
        valid,
        key_valid,
        AttentionKind.BLOCK_STRUCTURED_SELF_ATTENTION,
    )
