"""Vocabulary row validation and copying shared by weight adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch

VOCABULARY_KEYS = frozenset({"token_embedding.weight", "vocabulary_head.weight", "vocabulary_head.bias"})

@dataclass(frozen=True)
class VocabularyMapping:
    """Describe which source vocabulary rows are copied or discarded.

    Copied rows must have unique target rows. ``semantics`` and discard reasons
    are optional report labels and do not affect the tensor transformation.
    """

    source_to_target: Mapping[int, int]
    semantics: Mapping[int, str] = field(default_factory=dict)
    discarded_source: Mapping[int, str] = field(default_factory=dict)

    def validate(self, source_size: int, target_size: int) -> None:
        if (any(type(row) is not int for row in self.source_to_target)
                or any(type(row) is not int for row in self.source_to_target.values())
                or any(type(row) is not int for row in self.discarded_source)):
            raise TypeError("vocabulary row IDs must be integers")
        mapped_rows = set(self.source_to_target)
        discarded_rows = set(self.discarded_source)
        if mapped_rows & discarded_rows:
            raise ValueError("a source vocabulary row cannot be mapped and discarded")
        if mapped_rows | discarded_rows != set(range(source_size)):
            raise ValueError("every source vocabulary row must be mapped or explicitly discarded")
        targets = list(self.source_to_target.values())
        if len(set(targets)) != len(targets):
            raise ValueError("vocabulary mapping target rows must be unique")
        for target in self.source_to_target.values():
            if not 0 <= target < target_size:
                raise ValueError("target vocabulary row out of range: %d" % target)

    @classmethod
    def identity(cls, size: int) -> "VocabularyMapping":
        return cls(source_to_target={row: row for row in range(size)})


def copy_vocab_tensor(
    source: torch.Tensor,
    target: torch.Tensor,
    mapping: VocabularyMapping,
) -> torch.Tensor:
    """Copy rows after the caller validates vocabulary shapes and the mapping."""
    if source.shape == target.shape and all(
        source_row == target_row for source_row, target_row in mapping.source_to_target.items()
    ) and len(mapping.source_to_target) == source.shape[0]:
        return source.detach().to(device=target.device, dtype=target.dtype)
    result = target.detach().clone()
    rows = sorted(mapping.source_to_target.items())
    source_rows = torch.tensor([row[0] for row in rows], dtype=torch.long, device=source.device)
    target_rows = torch.tensor([row[1] for row in rows], dtype=torch.long, device=target.device)
    selected = source.detach().index_select(0, source_rows).to(device=target.device, dtype=target.dtype)
    result.index_copy_(0, target_rows, selected)
    return result
