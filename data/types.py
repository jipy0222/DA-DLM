"""Shared contracts for map-style data and collated token batches.

Dataset access is structural: both PyTorch datasets and Arrow datasets expose
integer indexing and a finite length. Values crossing dataset/JSON boundaries
remain open mappings; the numerical training batch has a fixed tensor schema.
"""

from __future__ import annotations

from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Mapping,
    Protocol,
    TypedDict,
    TypeVar,
    Union,
)

if TYPE_CHECKING:
    from torch import Tensor


Pathish = Union[str, Path]
DataRow = Mapping[str, Any]
CollateFn = Callable[[list[DataRow]], Any]
Metadata = dict[str, Any]
_Row = TypeVar("_Row", covariant=True)


class MapDataset(Protocol[_Row]):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int, /) -> _Row: ...


class _OptionalTokenBatch(TypedDict, total=False):
    loss_mask: Tensor
    sample_weight: Tensor


class TokenBatch(_OptionalTokenBatch):
    input_ids: Tensor
