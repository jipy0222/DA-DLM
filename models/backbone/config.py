"""Typed, local configuration for the shared Transformer backbone."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class AttentionBackend(str, Enum):
    SDPA = "sdpa"
    FLEX = "flex"


class AttentionKind(str, Enum):
    CAUSAL = "causal"
    BIDIRECTIONAL = "bidirectional"
    BLOCK_STRUCTURED_SELF_ATTENTION = "block_structured_self_attention"


class ConditioningMode(str, Enum):
    NONE = "none"
    ZERO_TIME_COMPAT = "zero_time_compat"
    ACTUAL_TIME = "actual_time"


class TransitionNormalization(str, Enum):
    LOCAL = "local"
    RAW_POTENTIAL = "raw_potential"


@dataclass(frozen=True)
class TransformerConfig:
    """Model settings; validation covers structural compatibility only."""

    hidden_size: int
    n_heads: int
    n_layers: int
    cond_dim: Optional[int] = None
    dropout: float = 0.1
    mlp_ratio: int = 4
    rope_base: float = 10_000.0
    max_attention_length: int = 1024
    max_position_id: float = 1023.0
    trained_context_length: int = 1024
    ada_ln: bool = True

    def __post_init__(self) -> None:
        if self.n_heads <= 0 or self.hidden_size % self.n_heads != 0:
            raise ValueError("n_heads must divide hidden_size")
        if (self.hidden_size // self.n_heads) % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        if self.mlp_ratio != 4:
            raise ValueError("TransformerBlock requires mlp_ratio=4")
        if self.ada_ln and self.cond_dim is None:
            raise ValueError("ada_ln=True requires cond_dim")


@dataclass(frozen=True)
class TokenFamilyConfig:
    """Shared network settings; public family configs validate enum choices."""

    backbone: TransformerConfig
    vocab_size: int
    pad_token_id: int
    conditioning: ConditioningMode
    default_backend: AttentionBackend
    tie_word_embeddings: bool = False
    timestep_embedding_size: int = 256
    # Total ordinary content, including the prompt; None uses training length.
    max_generation_length: Optional[int] = None

    def __post_init__(self) -> None:
        if not 0 <= self.pad_token_id < self.vocab_size:
            raise ValueError("pad_token_id must be inside the vocabulary")
        uses_conditioning = self.conditioning is not ConditioningMode.NONE
        if uses_conditioning != self.backbone.ada_ln:
            raise ValueError("conditioning and backbone.ada_ln must agree")
        if self.max_generation_length is not None:
            if not 0 < self.max_generation_length <= self.backbone.max_attention_length:
                raise ValueError("max_generation_length must fit the backbone attention capacity")
            if self.max_generation_length - 1 > self.backbone.max_position_id:
                raise ValueError("max_generation_length must fit the backbone RoPE positions")
