"""The family-agnostic Transformer stack."""

from __future__ import annotations

from typing import Optional, cast

import torch
from torch import nn

from kv_cache import CacheDelta, CacheMode, CacheRequest, KVCache, LayerCacheDelta
from kv_cache.packing import make_cache_packing_plan

from .attention import build_flex_block_mask
from .block import TransformerBlock
from .config import AttentionBackend, TransformerConfig
from .normalization import BiaslessLayerNorm, modulate, zero_linear_
from .rope import RotaryEmbedding, RotaryTables
from .types import AttentionLayout, BackboneOutput, PositionPolicy


class TransformerCore(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.rotary = RotaryEmbedding(
            config.hidden_size // config.n_heads, base=config.rope_base
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layers)
        )
        self.final_norm = BiaslessLayerNorm(config.hidden_size)
        if config.ada_ln:
            assert config.cond_dim is not None
            self.final_ada_ln = zero_linear_(
                nn.Linear(config.cond_dim, 2 * config.hidden_size, bias=True)
            )
        else:
            self.final_ada_ln = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        conditioning: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        attention_layout: AttentionLayout,
        backend: AttentionBackend = AttentionBackend.SDPA,
        position_policy: PositionPolicy = PositionPolicy(),
        cache_request: CacheRequest = CacheRequest(),
        output_hidden_states: bool = False,
    ) -> BackboneOutput:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, query, hidden]")
        batch, query_len, hidden_size = hidden_states.shape
        if hidden_size != self.config.hidden_size:
            raise ValueError("hidden_states has the wrong hidden dimension")
        if position_ids.shape != (batch, query_len):
            raise ValueError("position_ids must have shape [batch, query]")
        if position_ids.dtype is not torch.long:
            raise TypeError("position_ids must use torch.long")
        if torch.any(position_ids < 0):
            raise ValueError("position_ids must be non-negative")
        query_valid = attention_layout.query_valid
        if query_valid.shape != (batch, query_len):
            raise ValueError("attention_layout.query_valid must have shape [batch, query]")
        logical_positions = position_policy.logical_positions(position_ids)
        # Padding queries have no position budget: an inactive row or unused
        # tail of a DAG block must not reject otherwise in-range valid queries.
        # Mask the check only; keep caller positions and actual RoPE unchanged.
        logical_positions.masked_fill_(~query_valid, 0.0)
        if logical_positions.numel() and logical_positions.max().item() > self.config.max_position_id:
            raise ValueError("logical RoPE positions exceed max_position_id")
        if attention_layout.key_length > self.config.max_attention_length:
            raise ValueError("attention layout exceeds max_attention_length")
        if self.training and cache_request.mode is not CacheMode.NONE:
            raise ValueError("KV cache is inference-only")

        # Prepare backend data once for this relation and share it across layers.
        sdpa_mask = None
        flex_block_mask = None
        if backend is AttentionBackend.SDPA:
            sdpa_mask = attention_layout.as_sdpa_mask()
        elif backend is AttentionBackend.FLEX:
            if cache_request.mode is not CacheMode.NONE:
                raise ValueError("generation with FlexAttention is not supported")
            flex_block_mask = build_flex_block_mask(attention_layout)

        cache = cast(KVCache, cache_request.cache)
        packing_plan = None
        if cache_request.mode is not CacheMode.NONE:
            if len(cache.layers) != len(self.blocks):
                raise ValueError("cache layer count does not match core")
            if cache.batch_size != batch:
                raise ValueError("cache batch size does not match input")
            if cache.device != hidden_states.device:
                raise ValueError("cache and hidden_states must share a device")
            packing_plan = cache_request._packing_plan_for(query_valid)
            if packing_plan is None:
                packing_plan = make_cache_packing_plan(
                    cache.lengths, query_valid, width=attention_layout.key_length
                )
            if (
                attention_layout.key_valid is not packing_plan.valid
                and not torch.equal(attention_layout.key_valid, packing_plan.valid)
            ):
                raise ValueError("attention layout does not match packed cached K/V")

        # Derive tables once for these positions, then lazily cast once per Q/K
        # dtype used in this forward.
        rotary_cos_sin = RotaryTables(
            self.rotary.cos_sin(
                position_ids, position_policy, self.rotary.inv_freq.dtype
            )
        )
        collected = [hidden_states] if output_hidden_states else None
        layer_deltas = []
        for layer_index, block in enumerate(self.blocks):
            past_key = past_value = past_valid = None
            if cache is not None:
                past_key = cache.layers[layer_index].key
                past_value = cache.layers[layer_index].value
                past_valid = cache.valid
            hidden_states, current_key, current_value = block(
                hidden_states,
                conditioning=conditioning,
                rotary=self.rotary,
                position_ids=position_ids,
                position_policy=position_policy,
                attention_layout=attention_layout,
                backend=backend,
                past_key=past_key,
                past_value=past_value,
                past_valid=past_valid,
                flex_block_mask=flex_block_mask,
                rotary_cos_sin=rotary_cos_sin,
                packing_plan=packing_plan,
                sdpa_mask=sdpa_mask,
            )
            if cache_request.mode is CacheMode.STAGE_APPEND:
                layer_deltas.append(
                    LayerCacheDelta(
                        current_key.transpose(1, 2),
                        current_value.transpose(1, 2),
                    )
                )
            if collected is not None:
                collected.append(hidden_states)

        hidden_states = self.final_norm(hidden_states)
        if self.final_ada_ln is not None:
            if conditioning is None:
                raise ValueError("conditioned core requires conditioning")
            shift, scale = self.final_ada_ln(conditioning)[:, None, :].chunk(
                2, dim=-1
            )
            hidden_states = modulate(hidden_states, shift, scale)
        elif conditioning is not None:
            raise ValueError("unconditioned core received conditioning")

        cache_delta = None
        if cache_request.mode is CacheMode.STAGE_APPEND:
            cache_delta = CacheDelta(
                layers=tuple(layer_deltas),
                valid=query_valid,
                position_ids=position_ids,
                source_state_id=cache.state_id,
            )
        return BackboneOutput(
            last_hidden_state=hidden_states,
            cache_delta=cache_delta,
            hidden_states=None if collected is None else tuple(collected),
        )
