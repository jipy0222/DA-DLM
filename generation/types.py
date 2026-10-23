"""Small result and accounting types for token generation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

_PHASE_FIELDS = {
    "prefill": "prefill_nfe",
    "prediction": "prediction_nfe",
    "commit": "commit_nfe",
}


@dataclass
class GenerationStats:
    prediction_nfe: torch.Tensor
    prefill_nfe: torch.Tensor
    commit_nfe: torch.Tensor
    update_steps: torch.Tensor
    batch_calls: dict[str, int]

    @classmethod
    def empty(cls, batch_size: int, device: torch.device) -> "GenerationStats":
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        def zeros() -> torch.Tensor:
            return torch.zeros(batch_size, device=device, dtype=torch.long)
        return cls(zeros(), zeros(), zeros(), zeros(),
                   {phase: 0 for phase in _PHASE_FIELDS})

    def record(self, phase: str, rows: torch.Tensor) -> None:
        if phase not in _PHASE_FIELDS:
            raise ValueError(f"unknown generation phase: {phase}")
        if rows.dtype != torch.long or rows.ndim != 1:
            raise TypeError("rows must be a one-dimensional long tensor")
        if rows.numel() == 0:
            return
        target = getattr(self, _PHASE_FIELDS[phase])
        if rows.device != target.device:
            raise ValueError("rows and counters must share a device")
        target.index_add_(0, rows, torch.ones_like(rows))
        self.batch_calls[phase] += 1

    @property
    def total_nfe(self) -> torch.Tensor:
        return self.prefill_nfe + self.prediction_nfe + self.commit_nfe


@dataclass(frozen=True)
class GenerationResult:
    token_ids: torch.Tensor
    lengths: torch.Tensor
    finish_reason: tuple[str, ...]
    stats: GenerationStats
