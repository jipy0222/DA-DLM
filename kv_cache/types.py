"""Per-call cache requests and uncommitted K/V updates."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional, Tuple

from torch import Tensor

from .packing import CachePackingPlan

if TYPE_CHECKING:
    from .cache import KVCache


class CacheMode(str, Enum):
    NONE = "none"
    READ_ONLY = "read_only"
    STAGE_APPEND = "stage_append"


@dataclass(frozen=True)
class CacheRequest:
    mode: CacheMode = CacheMode.NONE
    cache: Optional[KVCache] = None
    _prepared_packing: Optional[Tuple[KVCache, Tensor, CachePackingPlan]] = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.mode, CacheMode):
            raise TypeError("mode must be a CacheMode")
        if self.mode is CacheMode.NONE:
            if self.cache is not None:
                raise ValueError("CacheMode.NONE cannot carry a cache")
        elif self.cache is None:
            raise ValueError("cached calls require cache data")

    def _packing_plan_for(self, current_valid: Tensor) -> Optional[CachePackingPlan]:
        """Reuse preparation only for its exact snapshot and borrowed query mask."""
        if self._prepared_packing is not None:
            cache, valid, plan = self._prepared_packing
            if cache is self.cache and valid is current_valid:
                return plan
        return None

    def prepared_packing(self) -> Tuple[Tensor, CachePackingPlan]:
        """Return this request's query mask and packing plan."""
        if self._prepared_packing is None:
            raise ValueError("cache request has no prepared packing")
        cache, valid, plan = self._prepared_packing
        if cache is None or cache is not self.cache:
            raise ValueError("prepared packing belongs to a different cache snapshot")
        return valid, plan


@dataclass(frozen=True)
class LayerCacheDelta:
    key: Tensor
    value: Tensor


@dataclass(frozen=True)
class CacheDelta:
    layers: Tuple[LayerCacheDelta, ...]
    valid: Tensor
    position_ids: Tensor
    source_state_id: object
