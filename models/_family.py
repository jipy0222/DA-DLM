"""Private composition helpers used by the four public family wrappers."""

from __future__ import annotations

from typing import Optional, cast

import torch
from torch import nn

from kv_cache import CacheMode, CacheRequest, CacheSession, KVCache

from .backbone.conditioning import TimestepConditioner, build_conditioning
from .backbone.config import ConditioningMode, TokenFamilyConfig
from .backbone.core import TransformerCore
from .backbone.embedding import TokenEmbedding
from .backbone.heads import VocabularyHead


class TokenFamilyModel(nn.Module):
    """Common ownership only; subclasses still define all family semantics."""

    def __init__(self, config: TokenFamilyConfig) -> None:
        super().__init__()
        self._cache_owner = object()
        self.family_config = config
        self.token_embedding = TokenEmbedding(
            config.vocab_size, config.backbone.hidden_size
        )
        if config.conditioning is ConditioningMode.NONE:
            self.conditioner = None
        else:
            self.conditioner = TimestepConditioner(
                cast(int, config.backbone.cond_dim),
                frequency_embedding_size=config.timestep_embedding_size,
            )
        self.core = TransformerCore(config.backbone)
        tied_weight = (
            self.token_embedding.weight if config.tie_word_embeddings else None
        )
        self.vocabulary_head = VocabularyHead(
            config.backbone.hidden_size,
            config.vocab_size,
            tied_weight=tied_weight,
        )

    def conditioning(
        self,
        sigma: Optional[torch.Tensor],
        *,
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        mode = self.family_config.conditioning
        if mode is ConditioningMode.NONE:
            if sigma is not None:
                raise ValueError("this family does not accept sigma")
            return None
        if sigma is None:
            if mode is ConditioningMode.ACTUAL_TIME:
                raise ValueError("ACTUAL_TIME requires sigma")
            sigma = torch.zeros(batch_size, device=device)
        if sigma.shape != (batch_size,):
            raise ValueError("sigma must have shape [batch]")
        if sigma.device != device:
            raise ValueError("sigma and token IDs must share a device")
        return build_conditioning(cast(TimestepConditioner, self.conditioner), mode, sigma)

    @property
    def _content_limit(self) -> int:
        """Total ordinary content capacity, including the prompt."""
        config = self.family_config
        limit = config.max_generation_length
        return config.backbone.trained_context_length if limit is None else limit

    def new_cache_session(
        self,
        batch_size: int,
        *,
        row_ids: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> CacheSession:
        parameter = self.token_embedding.weight
        if device is None:
            device = parameter.device
        if dtype is None:
            dtype = parameter.dtype
        head_dim = self.family_config.backbone.hidden_size // self.family_config.backbone.n_heads
        cache = KVCache.empty(
            batch_size=batch_size,
            n_layers=self.family_config.backbone.n_layers,
            n_heads=self.family_config.backbone.n_heads,
            head_dim=head_dim,
            device=device,
            dtype=dtype,
            model_owner=self._cache_owner,
            max_content_length=self._content_limit,
            row_ids=row_ids,
        )
        return CacheSession(cache)

    def cache_request(
        self,
        session: CacheSession,
        mode: CacheMode,
        budget_valid: torch.Tensor,
        *,
        packing_valid: Optional[torch.Tensor] = None,
    ) -> CacheRequest:
        """Prepare a cache request bound to this model instance."""
        if session.cache.model_owner is not self._cache_owner:
            raise ValueError("cache session belongs to a different model instance")
        return session.request(mode, budget_valid, packing_valid=packing_valid)


def pad_source_boundary(
    context_ids: torch.Tensor,
    context_position_ids: Optional[torch.Tensor],
    *,
    source_length: int,
    padding: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Insert PAD between source and target, never inside the target stream.

    Callers derive padding from source SLOT width and check total capacity and
    original position shape first. Existing positions remain unchanged; inserted
    PAD positions are irrelevant. The aligned case returns the original tensors.
    """
    if padding == 0:
        return context_ids, context_position_ids
    pad_ids = context_ids.new_full((context_ids.shape[0], padding), pad_token_id)
    context_ids = torch.cat(
        (context_ids[:, :source_length], pad_ids, context_ids[:, source_length:]), dim=1
    )
    if context_position_ids is not None:
        pad_positions = context_position_ids.new_zeros((context_ids.shape[0], padding))
        context_position_ids = torch.cat(
            (context_position_ids[:, :source_length], pad_positions,
             context_position_ids[:, source_length:]), dim=1
        )
    return context_ids, context_position_ids
