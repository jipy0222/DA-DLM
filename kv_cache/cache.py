"""Exact, dynamic, right-padded reference KV cache."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import Optional, Tuple

import torch

from .packing import make_cache_packing_plan
from .types import CacheDelta


class ContextLimitError(ValueError):
    def __init__(
        self,
        row_ids: torch.Tensor,
        attempted_lengths: torch.Tensor,
        limit: int,
    ) -> None:
        self.row_ids = row_ids.detach().clone()
        self.attempted_lengths = attempted_lengths.detach().clone()
        self.limit = limit
        super().__init__(
            "context limit %d exceeded by rows %s with attempted lengths %s"
            % (limit, self.row_ids.tolist(), self.attempted_lengths.tolist())
        )


@dataclass(frozen=True)
class LayerKV:
    key: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class KVCache:
    """A read-only-by-contract snapshot; frozen does not freeze Tensor storage.

    Callers must not mutate borrowed tensors, including metadata. Recreate the
    session after changing model weights, position policy or execution settings.
    Snapshot identity protects delta provenance, not in-place writes or weights.
    The content limit counts valid tokens, including the prompt.
    """

    layers: Tuple[LayerKV, ...]
    valid: torch.Tensor
    position_ids: torch.Tensor
    lengths: torch.Tensor
    row_ids: torch.Tensor
    model_owner: object
    max_content_length: int
    # Runtime identity of this exact snapshot.
    # Every constructed/replaced snapshot gets a fresh token; keeping only the
    # token in deltas does not retain the historical K/V tensors.
    state_id: object = field(default_factory=object, init=False, repr=False, compare=False)
    # Internal append/select_rows preserve the invariants. InitVar is not stored,
    # so ordinary construction and dataclasses.replace still validate the result.
    _validated: InitVar[bool] = False

    def __post_init__(self, _validated: bool) -> None:
        if _validated:
            return
        if not self.layers:
            raise ValueError("cache must contain at least one layer")
        if self.valid.dtype is not torch.bool or self.valid.ndim != 2:
            raise TypeError("cache valid must be boolean [batch, sequence]")
        batch, width = self.valid.shape
        if self.position_ids.shape != (batch, width):
            raise ValueError("cache position_ids shape mismatch")
        if self.position_ids.dtype is not torch.long:
            raise TypeError("cache position_ids must be torch.long")
        if self.lengths.shape != (batch,) or self.lengths.dtype is not torch.long:
            raise TypeError("cache lengths must be torch.long [batch]")
        if self.row_ids.shape != (batch,) or self.row_ids.dtype is not torch.long:
            raise TypeError("cache row_ids must be torch.long [batch]")
        if self.max_content_length <= 0:
            raise ValueError("max_content_length must be positive")
        if self.position_ids.device != self.valid.device:
            raise ValueError("cache metadata must share a device")
        if self.lengths.device != self.valid.device or self.row_ids.device != self.valid.device:
            raise ValueError("cache metadata must share a device")

        first_shape = self.layers[0].key.shape
        if len(first_shape) != 4 or first_shape[0] != batch or first_shape[2] != width:
            raise ValueError("layer K/V must have shape [batch, heads, sequence, head_dim]")
        first_dtype = self.layers[0].key.dtype
        first_device = self.layers[0].key.device
        for layer in self.layers:
            if layer.key.shape != first_shape or layer.value.shape != first_shape:
                raise ValueError("all layer K/V tensors must share a shape")
            if layer.key.dtype != first_dtype or layer.value.dtype != first_dtype:
                raise ValueError("all layer K/V tensors must share a dtype")
            if layer.key.device != first_device or layer.value.device != first_device:
                raise ValueError("all layer K/V tensors must share a device")
            if layer.key.requires_grad or layer.value.requires_grad:
                raise ValueError("committed cache tensors cannot require gradients")
        if first_device != self.valid.device:
            raise ValueError("cache K/V and metadata must share a device")
        self._validate_contents()

    def _validate_contents(self) -> None:
        """Tensor-value checks at the external construction boundary only."""
        if torch.unique(self.row_ids).numel() != self.batch_size:
            raise ValueError("cache row_ids must be unique")
        # Required for trusted packing: a length must refer to stored history,
        # not an out-of-storage or negative prefix that merely matches valid.
        if torch.any((self.lengths < 0) | (self.lengths > self.width)):
            raise ValueError("cache lengths must be between zero and physical width")
        expected_valid = torch.arange(self.width, device=self.device)[None] < self.lengths[:, None]
        if not torch.equal(self.valid, expected_valid):
            raise ValueError("cache must be physically right padded")
        if torch.any(self.lengths > self.max_content_length):
            raise ValueError("cache length exceeds max_content_length")

    @classmethod
    def empty(
        cls,
        *,
        batch_size: int,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        model_owner: object,
        max_content_length: int,
        row_ids: Optional[torch.Tensor] = None,
    ) -> "KVCache":
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if row_ids is None:
            row_ids = torch.arange(batch_size, device=device, dtype=torch.long)
        empty_kv = tuple(
            LayerKV(
                torch.empty(batch_size, n_heads, 0, head_dim, device=device, dtype=dtype),
                torch.empty(batch_size, n_heads, 0, head_dim, device=device, dtype=dtype),
            )
            for _ in range(n_layers)
        )
        return cls(
            layers=empty_kv,
            valid=torch.empty(batch_size, 0, device=device, dtype=torch.bool),
            position_ids=torch.empty(batch_size, 0, device=device, dtype=torch.long),
            lengths=torch.zeros(batch_size, device=device, dtype=torch.long),
            row_ids=row_ids.clone(),
            model_owner=model_owner,
            max_content_length=max_content_length,
        )

    @property
    def batch_size(self) -> int:
        return self.valid.shape[0]

    @property
    def width(self) -> int:
        return self.valid.shape[1]

    @property
    def dtype(self) -> torch.dtype:
        return self.layers[0].key.dtype

    @property
    def device(self) -> torch.device:
        return self.valid.device

    def preflight(self, append_valid: torch.Tensor) -> torch.Tensor:
        if append_valid.dtype is not torch.bool:
            raise TypeError("append_valid must be boolean")
        if append_valid.ndim != 2 or append_valid.shape[0] != self.batch_size:
            raise ValueError("append_valid must have shape [cache batch, append width]")
        if append_valid.device != self.device:
            raise ValueError("append_valid and cache must share a device")
        attempted = self.lengths + append_valid.sum(dim=-1)
        overflow = attempted > self.max_content_length
        if torch.any(overflow):
            raise ContextLimitError(
                self.row_ids[overflow], attempted[overflow], self.max_content_length
            )
        return attempted

    def append(self, delta: CacheDelta) -> "KVCache":
        if delta.source_state_id is not self.state_id:
            raise ValueError("cache delta source state mismatch")
        if len(delta.layers) != len(self.layers):
            raise ValueError("cache delta layer count mismatch")
        if delta.position_ids.device != self.device:
            raise ValueError("cache delta metadata device mismatch")
        if delta.position_ids.dtype is not torch.long:
            raise TypeError("cache delta position_ids must be torch.long")
        if delta.position_ids.shape != delta.valid.shape:
            raise ValueError("cache delta position metadata shape mismatch")
        # preflight owns validity metadata and capacity checks, including rank.
        new_lengths = self.preflight(delta.valid)
        new_width = int(new_lengths.max().item())
        plan = make_cache_packing_plan(
            self.lengths, delta.valid, width=new_width, _lengths=new_lengths
        )
        new_valid = plan.valid
        new_positions = plan.merge(
            self.position_ids[:, None, :, None],
            delta.position_ids[:, None, :, None],
        )[:, 0, :, 0]
        new_layers = []
        for layer_index, (old_layer, delta_layer) in enumerate(
            zip(self.layers, delta.layers)
        ):
            expected = (
                self.batch_size,
                old_layer.key.shape[1],
                delta.valid.shape[1],
                old_layer.key.shape[3],
            )
            if delta_layer.key.shape != expected or delta_layer.value.shape != expected:
                raise ValueError("cache delta K/V shape mismatch at layer %d" % layer_index)
            if delta_layer.key.device != self.device or delta_layer.value.device != self.device:
                raise ValueError("cache delta K/V device mismatch")
            if delta_layer.key.dtype != self.dtype or delta_layer.value.dtype != self.dtype:
                raise ValueError("cache delta K/V dtype mismatch")
            key = plan.merge(old_layer.key, delta_layer.key.detach())
            value = plan.merge(old_layer.value, delta_layer.value.detach())
            new_layers.append(LayerKV(key, value))

        return KVCache(
            layers=tuple(new_layers),
            valid=new_valid,
            position_ids=new_positions,
            lengths=new_lengths,
            row_ids=self.row_ids.clone(),
            model_owner=self.model_owner,
            max_content_length=self.max_content_length,
            _validated=True,
        )

    def select_rows(self, indices: torch.Tensor) -> "KVCache":
        self._validate_row_indices(indices)
        # CUDA index errors can invalidate the device context, unlike CPU errors.
        if indices.is_cuda and torch.any((indices < 0) | (indices >= self.batch_size)):
            raise IndexError("row index out of range")
        if torch.unique(indices).numel() != indices.numel():
            raise ValueError("row selection cannot duplicate stable row IDs")
        return self._select_rows(indices)

    def _validate_row_indices(self, indices: torch.Tensor) -> None:
        if indices.dtype is not torch.long or indices.ndim != 1:
            raise TypeError("row indices must be a one-dimensional long tensor")
        if indices.device != self.device:
            raise ValueError("row indices and cache must share a device")
        if indices.numel() == 0:
            raise ValueError("select_rows cannot create an empty cache")

    def _select_rows(self, indices: torch.Tensor) -> "KVCache":
        """Copy unique rows; index_select checks index bounds."""
        lengths = self.lengths.index_select(0, indices)
        width = int(lengths.max().item())
        layers = tuple(
            LayerKV(
                layer.key[:, :, :width].index_select(0, indices),
                layer.value[:, :, :width].index_select(0, indices),
            )
            for layer in self.layers
        )
        return KVCache(
            layers=layers,
            # index_select already owns its storage; slicing FIRST avoids
            # copying dropped tail slots and a second full clone afterward.
            valid=self.valid[:, :width].index_select(0, indices),
            position_ids=self.position_ids[:, :width].index_select(0, indices),
            lengths=lengths,
            row_ids=self.row_ids.index_select(0, indices),
            model_owner=self.model_owner,
            max_content_length=self.max_content_length,
            _validated=True,
        )
