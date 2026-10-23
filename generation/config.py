"""Small, task-independent generation configurations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from graph.runtime import GraphBackend

_STRATEGIES = frozenset({
    "default", "first_hitting", "confidence", "confidence_threshold",
    "sub_block", "nearby", "consecutive",
})
_STOP_MODES = frozenset({"none", "after_block", "trim_suffix"})


def _token_ids(ids: tuple[int, ...], name: str) -> None:
    if not isinstance(ids, tuple) or any(
        isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
        for token_id in ids
    ):
        raise ValueError(f"{name} must be a tuple of nonnegative token IDs")


@dataclass(frozen=True)
class SamplingConfig:
    greedy: bool = False
    top_k: int = 0
    top_p: float = 1.0
    blocked_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 0:
            raise ValueError("top_k must be a nonnegative integer")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k > 0 and self.top_p < 1:
            raise ValueError("top_k and top_p cannot be enabled together")
        _token_ids(self.blocked_token_ids, "blocked_token_ids")


@dataclass(frozen=True)
class StopConfig:
    mode: Literal["none", "after_block", "trim_suffix"] = "none"
    token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in _STOP_MODES:
            raise ValueError(f"unknown stop mode: {self.mode}")
        _token_ids(self.token_ids, "stop token_ids")
        if self.mode != "none" and not self.token_ids:
            raise ValueError("an enabled stop mode requires token_ids")


@dataclass(frozen=True)
class DADLMGenerationConfig:
    """Path decoding, confidence weighting and the graph execution backend."""

    decoder: Literal["ffbs", "viterbi"] = "ffbs"
    confidence_transition_weight: float = 0.5
    graph_backend: GraphBackend = "auto"

    def __post_init__(self) -> None:
        if self.decoder not in {"ffbs", "viterbi"}:
            raise ValueError("decoder must be ffbs or viterbi")
        if self.graph_backend not in {"auto", "pytorch", "cuda"}:
            raise ValueError("graph_backend must be auto, pytorch or cuda")
        if not math.isfinite(self.confidence_transition_weight) or not 0 <= self.confidence_transition_weight <= 1:
            raise ValueError("confidence_transition_weight must be in [0, 1]")


@dataclass(frozen=True)
class DiffusionConfig:
    num_steps: int
    strategy: str = "default"
    confidence_threshold: float = 0.9
    fallback_k: int = 1
    confidence_temperature: float = 0.0
    nearby_unmask_lambda: float = 1.0
    nearby_decay: float = 0.0
    remask_rho: float = 0.0
    infilling_k: int = 0
    infilling_min_context: float = 0.0
    infilling_threshold: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.num_steps, bool) or not isinstance(self.num_steps, int) or self.num_steps <= 0:
            raise ValueError("num_steps must be a positive integer")
        if self.strategy not in _STRATEGIES:
            raise ValueError(f"unknown reveal strategy: {self.strategy}")
        if isinstance(self.fallback_k, bool) or not isinstance(self.fallback_k, int) or self.fallback_k < 0:
            raise ValueError("fallback_k must be a nonnegative integer")
        if isinstance(self.infilling_k, bool) or not isinstance(self.infilling_k, int) or self.infilling_k < 0:
            raise ValueError("infilling_k must be a nonnegative integer")
        for name in ("confidence_threshold", "infilling_min_context", "infilling_threshold"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("confidence_temperature", "nearby_unmask_lambda"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
        if not math.isfinite(self.nearby_decay) or not 0 <= self.nearby_decay <= 1:
            raise ValueError("nearby_decay must be in [0, 1]")
        if not math.isfinite(self.remask_rho) or self.remask_rho > 1:
            raise ValueError("remask_rho must be finite and at most 1")
