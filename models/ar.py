"""Autoregressive family adapter over the shared Transformer core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, cast, overload

import torch

from kv_cache import CacheMode, CacheSession

from ._family import TokenFamilyModel
from .backbone.config import (
    AttentionBackend,
    AttentionKind,
    ConditioningMode,
    TokenFamilyConfig,
    TransformerConfig,
)
from .backbone.layouts import (
    cached_generation_layout,
    causal_layout,
    positions_from_valid,
)
from .backbone.types import BackboneOutput, TokenEncodingOutput, TokenModelOutput


@dataclass(frozen=True)
class ARConfig:
    backbone: TransformerConfig
    vocab_size: int
    pad_token_id: int
    tie_word_embeddings: bool = False
    default_backend: AttentionBackend = AttentionBackend.SDPA
    max_generation_length: Optional[int] = None

    def __post_init__(self) -> None:
        if self.backbone.ada_ln:
            raise ValueError("AR requires ada_ln=False")
        if self.default_backend is not AttentionBackend.SDPA:
            raise ValueError("AR default_backend must be AttentionBackend.SDPA")

    def shared(self) -> TokenFamilyConfig:
        return TokenFamilyConfig(
            backbone=self.backbone,
            vocab_size=self.vocab_size,
            pad_token_id=self.pad_token_id,
            conditioning=ConditioningMode.NONE,
            default_backend=self.default_backend,
            tie_word_embeddings=self.tie_word_embeddings,
            max_generation_length=self.max_generation_length,
        )


class ARModel(TokenFamilyModel):
    def __init__(self, config: ARConfig) -> None:
        super().__init__(config.shared())
        self.config = config

    def encode(
        self,
        input_ids: torch.Tensor,
        *,
        output_hidden_states: bool = False,
    ) -> BackboneOutput:
        valid = input_ids.ne(self.config.pad_token_id)
        hidden = self.token_embedding(input_ids)
        layout = causal_layout(valid)
        return cast('BackboneOutput', self.core(
            hidden,
            conditioning=None,
            position_ids=positions_from_valid(valid),
            attention_layout=layout,
            backend=self.config.default_backend,
            output_hidden_states=output_hidden_states,
        ))

    def forward(self, input_ids: torch.Tensor) -> TokenModelOutput:
        output = self.encode(input_ids)
        return TokenModelOutput(
            logits=self.vocabulary_head(output.last_hidden_state),
            cache_delta=output.cache_delta,
        )

    @overload
    def generate_step(
        self, input_ids: torch.Tensor, session: CacheSession, *,
        cache_mode: CacheMode, logits_mode: Literal["all", "last_valid"] = "all",
    ) -> TokenModelOutput: ...

    @overload
    def generate_step(
        self, input_ids: torch.Tensor, session: CacheSession, *,
        cache_mode: CacheMode, logits_mode: Literal["none"],
    ) -> TokenEncodingOutput: ...

    @overload
    def generate_step(
        self, input_ids: torch.Tensor, session: CacheSession, *,
        cache_mode: CacheMode, logits_mode: Literal["all", "last_valid", "none"],
    ) -> TokenModelOutput | TokenEncodingOutput: ...

    def generate_step(
        self,
        input_ids: torch.Tensor,
        session: CacheSession,
        *,
        cache_mode: CacheMode,
        logits_mode: Literal["all", "last_valid", "none"] = "all",
    ) -> TokenModelOutput | TokenEncodingOutput:
        """Compute current tokens without implicitly committing their K/V.

        ``all`` preserves [B,S,V] output; ``last_valid`` returns [B,1,V]
        with per-row validity and original input-slot indices (-1 if absent).
        Inactive rows have zero placeholder logits and MUST NOT be sampled.
        ``none`` skips the vocabulary head (e.g. intermediate prompt chunks).
        None of these modes skips Transformer positions or changes cache delta.
        EOS detection, row removal and grad scope remain caller-owned.
        """
        if logits_mode not in ("all", "last_valid", "none"):
            raise ValueError("logits_mode must be all, last_valid, or none")
        if input_ids.shape[1] == 0:
            raise ValueError("AR generation requires at least one input slot")
        valid = input_ids.ne(self.config.pad_token_id)
        request = self.cache_request(session, cache_mode, valid, packing_valid=valid)
        cache = session.cache
        positions = cache.lengths[:, None] + valid.long().cumsum(dim=-1) - 1
        positions = positions.clamp_min(0)
        layout = cached_generation_layout(
            request=request,
            causal_current=True,
            kind=AttentionKind.CAUSAL,
        )
        output = self.core(
            self.token_embedding(input_ids),
            conditioning=None,
            position_ids=positions,
            attention_layout=layout,
            backend=AttentionBackend.SDPA,
            cache_request=request,
        )
        if logits_mode == "none":
            return TokenEncodingOutput(cache_delta=output.cache_delta)
        if logits_mode == "all":
            return TokenModelOutput(
                self.vocabulary_head(output.last_hidden_state), output.cache_delta,
                logits_valid=valid,
            )

        # Use original slot indices, not valid.sum()-1: left/internal padding
        # is permitted. Only active rows enter the large vocabulary projection.
        slots = torch.arange(valid.shape[1], device=valid.device)
        last = torch.where(valid, slots[None], -1).amax(dim=-1)
        rows = last.ge(0).nonzero(as_tuple=True)[0]
        selected = output.last_hidden_state[rows, last[rows]]
        projected = self.vocabulary_head(selected)
        logits = projected.new_zeros(valid.shape[0], 1, projected.shape[-1])
        logits = logits.index_copy(0, rows, projected[:, None])
        return TokenModelOutput(
            logits, output.cache_delta,
            logits_valid=last.ge(0)[:, None], logits_positions=last[:, None],
        )
