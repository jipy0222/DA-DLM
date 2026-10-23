"""Library-backed LR schedules for Hydra/Lightning.

Transformers implements constant curves; timm implements cosine, including its
warmup-prefix convention. Only timm's step interface is adapted. Return the
scheduler to Lightning with interval="step"; do not also step it manually.

For manual resume: construct optimizer AND scheduler first, then restore their
library state_dicts (optimizer before scheduler). Scheduler loading does not
restore optimizer LR. Recreate the same configuration and parameter groups.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union, cast

import torch
from timm.scheduler import CosineLRScheduler


@dataclass(frozen=True)
class LRScheduleConfig:
    """LR schedule presets in optimizer-update units.

    Cosine supports a fractional warmup boundary (e.g. 0.1 * max_steps).
    The constant-warmup preset uses an integer step count.
    """

    kind: Literal["constant", "constant_warmup", "cosine_warmup"] = "constant_warmup"
    warmup_steps: Union[int, float] = 2500
    total_steps: Optional[int] = None
    warmup_lr_init: float = 0.0
    min_lr: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in ("constant", "constant_warmup", "cosine_warmup"):
            raise ValueError("unknown LR schedule kind")
        if not math.isfinite(self.warmup_steps) or self.warmup_steps < 0:
            raise ValueError("warmup_steps must be finite and nonnegative")
        if self.kind != "cosine_warmup":
            operator.index(cast(int, self.warmup_steps))
        if any(not math.isfinite(rate) or rate < 0
               for rate in (self.warmup_lr_init, self.min_lr)):
            raise ValueError("LR endpoints must be finite and nonnegative")
        if self.kind == "cosine_warmup":
            if self.total_steps is None or operator.index(self.total_steps) <= self.warmup_steps:
                raise ValueError("cosine total_steps must exceed warmup_steps")
        elif (self.total_steps is not None or self.warmup_lr_init != 0
              or self.min_lr != 0):
            raise ValueError("total_steps and absolute LR endpoints are cosine-only")
        if self.kind == "constant" and self.warmup_steps != 0:
            raise ValueError("constant schedule requires warmup_steps=0")


class CosineDecayWarmupLRScheduler(
        CosineLRScheduler, torch.optim.lr_scheduler.LRScheduler):
    """Adapt timm's step interface to Lightning.

    timm owns the curve and state serialization. The _last_epoch counter
    lets Lightning call step() without an explicit update index. The PyTorch
    base marks this as a supported scheduler; its constructor/step are not used.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._last_epoch = -1
        self.step(epoch=0)

    # Lightning uses the PyTorch step interface, without timm's metric argument.
    def step(self, epoch: int | None = None) -> None:  # ty: ignore[invalid-method-override]
        update = self._last_epoch + 1 if epoch is None else operator.index(epoch)
        if update < 0:
            raise ValueError("epoch must be nonnegative")
        self._last_epoch = update
        if self.t_in_epochs:
            super().step(epoch=self._last_epoch)
        else:
            super().step_update(num_updates=self._last_epoch)

    def get_last_lr(self) -> list[float]:
        return [group["lr"] for group in self.optimizer.param_groups]


def build_lr_scheduler(optimizer: torch.optim.Optimizer,
                       config: LRScheduleConfig) -> torch.optim.lr_scheduler.LRScheduler:
    """Create an upstream scheduler; Hydra can instantiate this factory.

    Parameter-group names, lr_scale and state loading follow library behavior.
    """
    for group in optimizer.param_groups:
        for name in ("lr", "initial_lr"):
            if name in group and (not math.isfinite(group[name]) or group[name] < 0):
                raise ValueError("%s must be finite and nonnegative" % name)
    if config.kind == "cosine_warmup":
        warmup_steps = float(config.warmup_steps)
        return CosineDecayWarmupLRScheduler(
            optimizer, t_initial=operator.index(cast(int, config.total_steps)) - warmup_steps,
            t_in_epochs=False, warmup_prefix=True, warmup_t=warmup_steps,
            warmup_lr_init=float(config.warmup_lr_init), lr_min=float(config.min_lr))
    from transformers import get_constant_schedule, get_constant_schedule_with_warmup

    if config.kind == "constant":
        return cast(torch.optim.lr_scheduler.LRScheduler, get_constant_schedule(optimizer))
    return cast(torch.optim.lr_scheduler.LRScheduler, get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=operator.index(cast(int, config.warmup_steps))))
