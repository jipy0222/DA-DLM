"""Block diffusion adapter over the shared Transformer core."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, cast

import torch

from kv_cache import CacheMode, CacheSession
from models.backbone.types import AttentionLayout

from ._family import TokenFamilyModel, pad_source_boundary
from .backbone.config import (
    AttentionBackend,
    AttentionKind,
    ConditioningMode,
    TokenFamilyConfig,
    TransformerConfig,
)
from .backbone.layouts import (
    block_causal_prefill_layout,
    block_structured_layout,
    cached_generation_layout,
    positions_from_valid,
)
from .backbone.types import BackboneOutput, TokenModelOutput


@dataclass(frozen=True)
class BD3LMConfig:
    backbone: TransformerConfig
    vocab_size: int
    pad_token_id: int
    block_size: int
    conditioning: ConditioningMode = ConditioningMode.ZERO_TIME_COMPAT
    tie_word_embeddings: bool = False
    default_backend: AttentionBackend = AttentionBackend.FLEX
    timestep_embedding_size: int = 256
    # Total valid content (prompt + completed blocks + current block), not
    # the doubled physical width of the full-sequence training layout.
    # None uses backbone.trained_context_length as the content limit.
    max_generation_length: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.conditioning, ConditioningMode):
            raise TypeError("conditioning must be a ConditioningMode")
        if not isinstance(self.default_backend, AttentionBackend):
            raise TypeError("default_backend must be an AttentionBackend")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.conditioning is ConditioningMode.NONE:
            raise ValueError("BD3LM requires AdaLN conditioning")
        if self.max_generation_length is not None and self.max_generation_length <= 0:
            raise ValueError("max_generation_length must be positive")

    def with_lengths(
        self,
        *,
        context_length: int,
        current_length: Optional[int] = None,
        generation_length: Optional[int] = None,
    ) -> "BD3LMConfig":
        """Return a new config with explicitly derived execution capacities.

        context_length is the clean source+target slot width; current_length
        is its noisy target suffix (defaults to the whole context). Generation
        length counts ALL valid content, including the prompt. If omitted, it
        uses max_generation_length, then trained_context_length. RoPE derivation
        assumes the default zero-based positions.

        Source boundary PAD is included in attention capacity, but does not
        spend position or generation budget. The returned config updates the
        two execution limits and preserves model dimensions and
        trained_context_length. The input config is unchanged.
        The two training widths size a shared capacity, not separate row limits.
        """
        if current_length is None:
            current_length = context_length
        if generation_length is None:
            generation_length = self.max_generation_length
        if generation_length is None:
            generation_length = self.backbone.trained_context_length
        if not 0 < current_length <= context_length or generation_length <= 0:
            raise ValueError("lengths require 0 < current_length <= context_length and generation_length > 0")
        source_padding = (current_length - context_length) % self.block_size
        return replace(
            self,
            backbone=replace(
                self.backbone,
                max_attention_length=max(context_length + source_padding + current_length, generation_length),
                max_position_id=max(context_length, generation_length) - 1,
            ),
            max_generation_length=generation_length,
        )

    def shared(self) -> TokenFamilyConfig:
        return TokenFamilyConfig(
            backbone=self.backbone,
            vocab_size=self.vocab_size,
            pad_token_id=self.pad_token_id,
            conditioning=self.conditioning,
            default_backend=self.default_backend,
            tie_word_embeddings=self.tie_word_embeddings,
            timestep_embedding_size=self.timestep_embedding_size,
            max_generation_length=self.max_generation_length,
        )


class BD3LMModel(TokenFamilyModel):
    def __init__(self, config: BD3LMConfig) -> None:
        super().__init__(config.shared())
        self.config = config

    def _inputs(
        self,
        current_ids: torch.Tensor,
        context_ids: torch.Tensor,
        current_position_ids: Optional[torch.Tensor],
        context_position_ids: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, AttentionLayout]:
        if current_ids.ndim != 2 or context_ids.ndim != 2:
            raise ValueError("current_ids and context_ids must have shape [batch, sequence]")
        if current_ids.shape[0] != context_ids.shape[0]:
            raise ValueError("current_ids and context_ids batch sizes must match")
        if current_ids.shape[1] > context_ids.shape[1]:
            raise ValueError("current sequence must align to a suffix of context")
        source_length = context_ids.shape[1] - current_ids.shape[1]
        source_padding = (-source_length) % self.config.block_size
        padded_context_length = context_ids.shape[1] + source_padding
        total_length = current_ids.shape[1] + padded_context_length
        if total_length > self.config.backbone.max_attention_length:
            raise ValueError(
                "BD3LM current + context width after source-boundary padding (%d + %d = %d) exceeds max_attention_length=%d; "
                "configure the two-stream capacity explicitly or use BD3LMConfig.with_lengths()"
                % (current_ids.shape[1], padded_context_length, total_length,
                   self.config.backbone.max_attention_length)
            )
        if context_position_ids is not None and context_position_ids.shape != context_ids.shape:
            raise ValueError("context_position_ids do not match context_ids")
        context_ids, context_position_ids = pad_source_boundary(
            context_ids, context_position_ids, source_length=source_length,
            padding=source_padding, pad_token_id=self.config.pad_token_id,
        )
        current_valid = current_ids.ne(self.config.pad_token_id)
        context_valid = context_ids.ne(self.config.pad_token_id)
        start = context_ids.shape[1] - current_ids.shape[1]
        if context_position_ids is None:
            context_position_ids = positions_from_valid(context_valid)
        if current_position_ids is None:
            current_position_ids = context_position_ids[:, start:]
        if current_position_ids.shape != current_ids.shape:
            raise ValueError("current_position_ids do not match current_ids")

        device = current_ids.device
        current_blocks = torch.arange(
            start, context_ids.shape[1], device=device
        ) // self.config.block_size
        context_blocks = torch.arange(
            context_ids.shape[1], device=device
        ) // self.config.block_size
        layout = block_structured_layout(
            current_valid=current_valid,
            context_valid=context_valid,
            current_block_ids=current_blocks,
            context_block_ids=context_blocks,
        )
        return (
            torch.cat((current_ids, context_ids), dim=1),
            torch.cat((current_position_ids, context_position_ids), dim=1),
            layout,
        )

    def encode(
        self,
        current_ids: torch.Tensor,
        context_ids: torch.Tensor,
        sigma: Optional[torch.Tensor] = None,
        *,
        current_position_ids: Optional[torch.Tensor] = None,
        context_position_ids: Optional[torch.Tensor] = None,
        backend: Optional[AttentionBackend] = None,
        output_hidden_states: bool = False,
    ) -> BackboneOutput:
        ids, positions, layout = self._inputs(
            current_ids,
            context_ids,
            current_position_ids,
            context_position_ids,
        )
        conditioning = self.conditioning(
            sigma, batch_size=ids.shape[0], device=ids.device
        )
        full = self.core(
            self.token_embedding(ids),
            conditioning=conditioning,
            position_ids=positions,
            attention_layout=layout,
            backend=self.config.default_backend if backend is None else backend,
            output_hidden_states=output_hidden_states,
        )
        length = current_ids.shape[1]
        hidden_states = None
        if full.hidden_states is not None:
            hidden_states = tuple(item[:, :length] for item in full.hidden_states)
        return BackboneOutput(
            full.last_hidden_state[:, :length],
            cache_delta=full.cache_delta,
            hidden_states=hidden_states,
        )

    def forward(
        self,
        current_ids: torch.Tensor,
        context_ids: torch.Tensor,
        sigma: Optional[torch.Tensor] = None,
        *,
        current_position_ids: Optional[torch.Tensor] = None,
        context_position_ids: Optional[torch.Tensor] = None,
        backend: Optional[AttentionBackend] = None,
    ) -> TokenModelOutput:
        output = self.encode(
            current_ids,
            context_ids,
            sigma,
            current_position_ids=current_position_ids,
            context_position_ids=context_position_ids,
            backend=backend,
        )
        return TokenModelOutput(
            self.vocabulary_head(output.last_hidden_state),
            output.cache_delta,
        )

    def prefill_prompt(
        self,
        prompt_ids: torch.Tensor,
        session: CacheSession,
        sigma: Optional[torch.Tensor] = None,
        *,
        block_ids: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> BackboneOutput:
        """Compute an external prompt delta; caller explicitly session.commit()s it.

        Requires empty history. Default blocks follow input slots; supply
        explicit block_ids for a different prompt layout.
        A BOS inside the first generation block instead goes to encode_generation.
        """

        if session.cache.width != 0:
            raise ValueError("prompt prefill requires an empty cache")
        valid = prompt_ids.ne(self.config.pad_token_id)
        request = self.cache_request(session, CacheMode.STAGE_APPEND, valid)
        if block_ids is None:
            block_ids = torch.arange(prompt_ids.shape[1], device=prompt_ids.device) // self.config.block_size
        return cast('BackboneOutput', self.core(
            self.token_embedding(prompt_ids),
            conditioning=self.conditioning(
                sigma, batch_size=prompt_ids.shape[0], device=prompt_ids.device
            ),
            position_ids=positions_from_valid(valid),
            attention_layout=block_causal_prefill_layout(valid=valid, block_ids=block_ids),
            backend=AttentionBackend.SDPA,
            cache_request=request,
            output_hidden_states=output_hidden_states,
        ))

    def encode_generation(
        self,
        current_ids: torch.Tensor,
        session: CacheSession,
        sigma: Optional[torch.Tensor] = None,
        *,
        cache_mode: CacheMode,
        output_hidden_states: bool = False,
    ) -> BackboneOutput:
        """Preview or stage ONE generation block, including any fixed BOS.

        Current tokens attend bidirectionally, regardless of history alignment
        or cache_mode. Use prefill_prompt for an external, possibly multi-block
        prompt. Commit the returned delta directly or stage it for a later flush.
        """

        valid = current_ids.ne(self.config.pad_token_id)
        request = self.cache_request(session, cache_mode, valid, packing_valid=valid)
        cache = session.cache
        positions = cache.lengths[:, None] + valid.long().cumsum(dim=-1) - 1
        positions = positions.clamp_min(0)
        layout = cached_generation_layout(
            request=request,
            causal_current=False,
            kind=AttentionKind.BLOCK_STRUCTURED_SELF_ATTENTION,
        )
        conditioning = self.conditioning(
            sigma, batch_size=current_ids.shape[0], device=current_ids.device
        )
        return cast('BackboneOutput', self.core(
            self.token_embedding(current_ids),
            conditioning=conditioning,
            position_ids=positions,
            attention_layout=layout,
            backend=AttentionBackend.SDPA,
            cache_request=request,
            output_hidden_states=output_hidden_states,
        ))

    def generation_step(
        self,
        current_ids: torch.Tensor,
        session: CacheSession,
        sigma: Optional[torch.Tensor] = None,
        *,
        cache_mode: CacheMode = CacheMode.READ_ONLY,
    ) -> TokenModelOutput:
        output = self.encode_generation(
            current_ids, session, sigma, cache_mode=cache_mode
        )
        return TokenModelOutput(
            self.vocabulary_head(output.last_hidden_state), output.cache_delta
        )
