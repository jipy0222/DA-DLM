"""Pre-norm Transformer block with self-attention and a GELU MLP."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, cast

import torch
from torch import nn

from kv_cache.packing import CachePackingPlan

from .attention import MultiheadSelfAttention
from .config import AttentionBackend, TransformerConfig
from .normalization import BiaslessLayerNorm, gated_residual, modulate, zero_linear_
from .rope import RotaryEmbedding, RotaryTableInput
from .types import AttentionLayout, PositionPolicy

if TYPE_CHECKING:
    from torch.nn.attention.flex_attention import BlockMask


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.ada_ln = config.ada_ln
        self.norm1 = BiaslessLayerNorm(config.hidden_size)
        self.attention = MultiheadSelfAttention(config)
        self.norm2 = BiaslessLayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, 4 * config.hidden_size, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * config.hidden_size, config.hidden_size, bias=True),
        )
        if config.ada_ln:
            assert config.cond_dim is not None
            self.ada_ln_modulation = zero_linear_(
                nn.Linear(config.cond_dim, 6 * config.hidden_size, bias=True)
            )
        else:
            self.ada_ln_modulation = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        conditioning: Optional[torch.Tensor],
        rotary: RotaryEmbedding,
        position_ids: torch.Tensor,
        position_policy: PositionPolicy,
        attention_layout: AttentionLayout,
        backend: AttentionBackend,
        past_key: Optional[torch.Tensor] = None,
        past_value: Optional[torch.Tensor] = None,
        past_valid: Optional[torch.Tensor] = None,
        flex_block_mask: Optional[BlockMask] = None,
        rotary_cos_sin: Optional[RotaryTableInput] = None,
        packing_plan: Optional[CachePackingPlan] = None,
        sdpa_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.ada_ln:
            if conditioning is None:
                raise ValueError("this block requires [batch, cond_dim] conditioning")
            if conditioning.ndim != 2 or conditioning.shape[0] != hidden_states.shape[0]:
                raise ValueError("conditioning must have shape [batch, cond_dim]")
            assert self.ada_ln_modulation is not None
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.ada_ln_modulation(conditioning)[:, None, :].chunk(6, dim=-1)
            )
            attention_input = modulate(
                self.norm1(hidden_states), shift_msa, scale_msa
            )
        else:
            if conditioning is not None:
                raise ValueError("unconditioned block received conditioning")
            gate_msa = gate_mlp = None
            shift_mlp = scale_mlp = None
            attention_input = self.norm1(hidden_states)

        attention = self.attention(
            attention_input,
            rotary=rotary,
            position_ids=position_ids,
            position_policy=position_policy,
            layout=attention_layout,
            backend=backend,
            past_key=past_key,
            past_value=past_value,
            past_valid=past_valid,
            flex_block_mask=flex_block_mask,
            rotary_cos_sin=rotary_cos_sin,
            packing_plan=packing_plan,
            sdpa_mask=sdpa_mask,
        )
        branch = self.dropout(attention.hidden_states)
        hidden_states = gated_residual(hidden_states, branch, gate_msa)

        mlp_input = self.norm2(hidden_states)
        if shift_mlp is not None and scale_mlp is not None:
            mlp_input = modulate(mlp_input, shift_mlp, scale_mlp)
        branch = self.dropout(self.mlp(mlp_input))
        hidden_states = gated_residual(hidden_states, branch, gate_mlp)
        return cast('tuple[torch.Tensor, torch.Tensor, torch.Tensor]', (hidden_states, attention.current_key, attention.current_value))
