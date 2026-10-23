"""Masked diffusion language-model adapter over the shared core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NoReturn, Optional, cast

import torch

from ._family import TokenFamilyModel
from .backbone.config import (
    AttentionBackend,
    ConditioningMode,
    TokenFamilyConfig,
    TransformerConfig,
)
from .backbone.layouts import bidirectional_layout, positions_from_valid
from .backbone.types import BackboneOutput, TokenModelOutput


@dataclass(frozen=True)
class MDLMConfig:
    backbone: TransformerConfig
    vocab_size: int
    pad_token_id: int
    conditioning: ConditioningMode = ConditioningMode.ZERO_TIME_COMPAT
    tie_word_embeddings: bool = False
    default_backend: AttentionBackend = AttentionBackend.SDPA
    timestep_embedding_size: int = 256
    max_generation_length: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.conditioning, ConditioningMode):
            raise TypeError("conditioning must be a ConditioningMode")
        if self.conditioning is ConditioningMode.NONE:
            raise ValueError("MDLM requires AdaLN conditioning")
        if self.default_backend is not AttentionBackend.SDPA:
            raise ValueError("MDLM default_backend must be AttentionBackend.SDPA")

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


class MDLMModel(TokenFamilyModel):
    def __init__(self, config: MDLMConfig) -> None:
        super().__init__(config.shared())
        self.config = config

    def encode(
        self,
        input_ids: torch.Tensor,
        sigma: Optional[torch.Tensor] = None,
        *,
        output_hidden_states: bool = False,
    ) -> BackboneOutput:
        valid = input_ids.ne(self.config.pad_token_id)
        conditioning = self.conditioning(
            sigma, batch_size=input_ids.shape[0], device=input_ids.device
        )
        return cast('BackboneOutput', self.core(
            self.token_embedding(input_ids),
            conditioning=conditioning,
            position_ids=positions_from_valid(valid),
            attention_layout=bidirectional_layout(valid),
            backend=self.config.default_backend,
            output_hidden_states=output_hidden_states,
        ))

    def forward(
        self,
        input_ids: torch.Tensor,
        sigma: Optional[torch.Tensor] = None,
    ) -> TokenModelOutput:
        output = self.encode(input_ids, sigma)
        return TokenModelOutput(
            logits=self.vocabulary_head(output.last_hidden_state),
            cache_delta=output.cache_delta,
        )

    def new_cache_session(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise ValueError(
            "MDLM regular KV cache is undefined because every denoising step changes the full sequence"
        )
