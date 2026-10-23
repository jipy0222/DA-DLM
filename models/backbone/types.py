"""Public value types shared by backbone modules and family adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, cast

import torch

from kv_cache import CacheDelta

from .config import AttentionKind, TransitionNormalization

Tensor = torch.Tensor


@dataclass(frozen=True)
class PositionPolicy:
    """How expanded integer position IDs are converted to RoPE coordinates."""

    scale: float = 1.0
    grouped: bool = False

    def __post_init__(self) -> None:
        if self.grouped:
            if self.scale <= 0:
                raise ValueError("grouped position scale must be a positive reciprocal integer")
            group_size = round(1.0 / self.scale)
            if group_size < 1 or self.scale != 1.0 / group_size:
                raise ValueError("grouped position scale must be a positive reciprocal integer")

    def logical_positions(self, position_ids: Tensor) -> Tensor:
        if self.grouped:
            # Divide integer node IDs before conversion to floating coordinates.
            group_size = round(1.0 / self.scale)
            return torch.div(position_ids, group_size, rounding_mode="floor").float()
        return position_ids.float() * self.scale


@dataclass(frozen=True, init=False)
class AttentionLayout:
    """Backend-independent declaration of permitted query-key relations.

    Supply either a key-masked ``allowed`` [B,Q,K], or a shared
    ``structural_allowed`` [Q,K] combined with ``key_valid`` by the backend.
    Construction checks metadata only. Reading ``allowed`` for a structural
    layout materializes the full relation; Flex deliberately does not do that.
    """

    _allowed: Optional[Tensor]
    query_valid: Tensor
    key_valid: Tensor
    kind: AttentionKind
    structural_allowed: Optional[Tensor]

    def __init__(
        self,
        allowed: Optional[Tensor],
        query_valid: Tensor,
        key_valid: Tensor,
        kind: AttentionKind,
        *,
        structural_allowed: Optional[Tensor] = None,
    ) -> None:
        object.__setattr__(self, "_allowed", allowed)
        object.__setattr__(self, "query_valid", query_valid)
        object.__setattr__(self, "key_valid", key_valid)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "structural_allowed", structural_allowed)
        self.__post_init__()

    def __post_init__(self) -> None:
        if (self._allowed is None) == (self.structural_allowed is None):
            raise ValueError("supply either allowed or structural_allowed")
        if self.query_valid.dtype is not torch.bool:
            raise TypeError("query_valid must be boolean")
        if self.key_valid.dtype is not torch.bool:
            raise TypeError("key_valid must be boolean")
        if self.query_valid.ndim != 2 or self.key_valid.ndim != 2:
            raise ValueError("validity tensors must have shape [batch, length]")
        if self.query_valid.shape[0] != self.key_valid.shape[0]:
            raise ValueError("layout batch sizes must match")
        relation = self._allowed if self._allowed is not None else cast(Tensor, self.structural_allowed)
        expected = (self.query_length, self.key_length)
        if self._allowed is not None:
            expected = (self.batch_size,) + expected
        if relation.dtype is not torch.bool or relation.shape != expected:
            raise ValueError("layout relation must be boolean with shape %s" % (expected,))
        if relation.device != self.query_valid.device or relation.device != self.key_valid.device:
            raise ValueError("layout tensors must share a device")

    @property
    def allowed(self) -> Tensor:
        """Return the explicit mask or combine shared structure with key validity."""
        if self._allowed is not None:
            return self._allowed
        return cast(Tensor, self.structural_allowed)[None] & self.key_valid[:, None, :]

    @property
    def batch_size(self) -> int:
        return self.query_valid.shape[0]

    @property
    def query_length(self) -> int:
        return self.query_valid.shape[1]

    @property
    def key_length(self) -> int:
        return self.key_valid.shape[1]

    def as_sdpa_mask(self) -> Tensor:
        """Prepare a mask once per core forward; layers reuse the returned view."""

        allowed = self.allowed
        if self.kind is AttentionKind.BIDIRECTIONAL:
            expected = self.key_valid[:, None, :].expand(
                -1, self.query_length, -1
            )
            if torch.equal(allowed, expected):
                return self.key_valid[:, None, None, :]
        return allowed[:, None, :, :]


@dataclass(frozen=True)
class BackboneOutput:
    last_hidden_state: Tensor
    cache_delta: Optional[CacheDelta] = None
    hidden_states: Optional[Tuple[Tensor, ...]] = None


@dataclass(frozen=True)
class TokenModelOutput:
    logits: Tensor
    cache_delta: Optional[CacheDelta] = None
    # Optional selection metadata for generation. None preserves the ordinary
    # full-sequence contract. Positions index THIS call's input slots, not RoPE.
    logits_valid: Optional[Tensor] = None
    logits_positions: Optional[Tensor] = None


@dataclass(frozen=True)
class TransitionOutput:
    scores: Tensor
    edge_valid: Tensor
    normalization: TransitionNormalization


@dataclass(frozen=True)
class DADModelOutput:
    logits: Tensor
    links: TransitionOutput
    cache_delta: Optional[CacheDelta] = None


@dataclass(frozen=True)
class TokenEncodingOutput:
    """Headless generation result; keep the same cache/selection attributes."""

    cache_delta: Optional[CacheDelta] = None
    logits: None = None
    logits_valid: None = None
    logits_positions: None = None
