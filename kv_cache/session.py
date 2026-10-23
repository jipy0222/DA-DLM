"""One batch's KV cache snapshot and pending update."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import torch

from .cache import KVCache
from .packing import make_cache_packing_plan
from .types import CacheDelta, CacheMode, CacheRequest, LayerCacheDelta


class CacheSession:
    """Single-owner state for one batch.

    Sampling callers use torch.no_grad() around model calls; eval() alone does
    not disable autograd. Pending K/V are detached but borrow their storage.
    Treat cache/delta tensors as read-only and discard this session after model
    weights or execution semantics change.
    """

    def __init__(self, cache: KVCache) -> None:
        self._cache = cache
        self._pending: Optional[CacheDelta] = None

    @property
    def cache(self) -> KVCache:
        return self._cache

    @property
    def has_pending(self) -> bool:
        return self._pending is not None

    def request(
        self,
        mode: CacheMode,
        budget_valid: torch.Tensor,
        *,
        packing_valid: Optional[torch.Tensor] = None,
    ) -> CacheRequest:
        """Check content capacity and optionally prepare this call's K/V packing.

        The budget mask counts content slots; the packing mask describes actual
        queries. The prepared plan borrows that mask and is shared across layers.
        """
        if self._pending is not None:
            raise ValueError("flush or discard the pending cache delta before generating")
        if mode is CacheMode.NONE:
            raise ValueError("a cache session cannot create CacheMode.NONE requests")
        lengths = self._cache.preflight(budget_valid)
        prepared = None
        if packing_valid is not None:
            plan = make_cache_packing_plan(
                self._cache.lengths,
                packing_valid,
                _lengths=lengths if packing_valid is budget_valid else None,
            )
            prepared = (self._cache, packing_valid, plan)
        return CacheRequest(mode, self._cache, prepared)

    def commit(self, delta: CacheDelta) -> None:
        if self._pending is not None:
            raise ValueError("flush or discard the pending cache delta first")
        self._cache = self._cache.append(delta)

    def stage(self, delta: CacheDelta) -> None:
        """Hold accepted K/V without owning their autograd graph or cloning.

        Detach breaks graph ownership, not storage aliasing: neither caller nor
        session may mutate the borrowed tensors. It does not make the caller's
        original delta/output nondifferentiable, nor release any live references
        held by the caller. flush() creates independently stored committed K/V.
        """

        if self._pending is not None:
            raise ValueError("a cache delta is already pending")
        if delta.source_state_id is not self._cache.state_id:
            raise ValueError("pending cache delta source state mismatch")
        self._pending = replace(
            delta,
            layers=tuple(
                LayerCacheDelta(layer.key.detach(), layer.value.detach())
                for layer in delta.layers
            ),
        )

    def flush(self) -> None:
        """Atomically commit the pending token/block; raise if none is staged."""

        if self._pending is None:
            raise ValueError("there is no pending cache delta to flush")
        self._cache = self._cache.append(self._pending)
        self._pending = None

    def discard_pending(self) -> None:
        self._pending = None

    def select_rows(self, indices: torch.Tensor) -> None:
        if self._pending is not None:
            raise ValueError("flush or discard the pending cache delta before selecting rows")
        self._cache = self._cache.select_rows(indices)

    def reorder_rows(self, permutation: torch.Tensor) -> None:
        if self._pending is not None:
            raise ValueError("flush or discard the pending cache delta before selecting rows")
        self._cache._validate_row_indices(permutation)
        if permutation.numel() != self._cache.batch_size:
            raise ValueError("reorder permutation must contain every current row")
        expected = torch.arange(
            self._cache.batch_size, device=permutation.device, dtype=torch.long
        )
        if not torch.equal(permutation.sort().values, expected):
            raise ValueError("reorder permutation must be a bijection")
        # A bijection establishes both range and uniqueness; do not scan twice.
        self._cache = self._cache._select_rows(permutation)
