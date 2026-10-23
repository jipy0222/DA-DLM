"""Shared token-weighted Lightning training lifecycle for token models."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from typing import Any, Generic, Optional, Sequence, TypeVar, Union, cast

import torch
from lightning.pytorch import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor

from models.backbone.types import DADModelOutput, TokenModelOutput
from objectives import LossTerms
from training._types import (
    Checkpoint,
    NativeModel,
    Pathish,
    TokenCheckpointState,
    TrainerCallbacks,
    TrainingBatch,
    TrainingContract,
    ValidationResults,
)
from training.ema import ExponentialMovingAverage
from training.lr_schedules import LRScheduleConfig, build_lr_scheduler


@dataclass(frozen=True)
class AdamWConfig:
    """AdamW optimizer settings."""

    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0


_Model = TypeVar("_Model", bound=NativeModel)


class TokenTrainingModule(LightningModule, Generic[_Model]):
    """Token-weighted optimization with optimizer-step EMA and LR schedules.

    Supports one AdamW optimizer, ordinary DDP or a single device, FP32 or
    BF16 mixed precision, and integer gradient accumulation. Backward uses
    loss sums; gradients are normalized over each actual global optimizer
    window before clipping, including incomplete final windows.

    The shared fit lifecycle restores optimizer, scheduler, EMA, metric and
    consumed-data state. It does not make dropout or stochastic-objective draws
    bitwise identical across restarts or distributed layouts.
    """

    checkpoint_key = "token_training"
    training_name = "token"

    def __init__(self, model: _Model, *, blocked_token_ids: Sequence[int],
                 optimizer: AdamWConfig = AdamWConfig(),
                 schedule: LRScheduleConfig = LRScheduleConfig(),
                 ema_decay: Optional[float] = 0.9999) -> None:
        super().__init__()
        self.model: _Model = model
        blocked = tuple(blocked_token_ids)
        vocab_size = model.config.vocab_size
        if (any(type(index) is not int or not 0 <= index < vocab_size for index in blocked)
                or len(set(blocked)) != len(blocked) or len(blocked) == vocab_size):
            raise ValueError("blocked token IDs must be unique vocabulary IDs and leave an output class")
        self.blocked_token_ids = tuple(sorted(blocked))
        self.optimizer_config = optimizer
        self.schedule_config = schedule
        self.ema: ExponentialMovingAverage | None = (None if ema_decay is None else
                    ExponentialMovingAverage(model, decay=ema_decay))
        self._window_count: Tensor | None = None
        self._train_totals: Tensor | None = None
        self._train_totals_epoch: int | None = None
        self._resume_train_totals: Tensor | None = None
        self._valid_totals: Tensor | None = None
        self._valid_metric_totals: dict[str, Tensor] = {}
        self._validation_in_progress = False
        self._ema_selection: AbstractContextManager[None] | None = None
        self._ema_ready = False
        self.last_train_metrics: dict[str, Tensor] = {}
        self.last_validation_metrics: dict[str, Tensor] = {}

    def forward(self, *args: Any, **kwargs: Any) -> TokenModelOutput | DADModelOutput:
        return cast(Union[TokenModelOutput, DADModelOutput], self.model(*args, **kwargs))

    def _terms(self, batch: TrainingBatch) -> LossTerms:
        raise NotImplementedError

    def _validation_terms(self, batch: TrainingBatch) -> LossTerms:
        return self._terms(batch)

    def _sum_ranks(self, tensor: Tensor) -> Tensor:
        return cast(Tensor, self.trainer.strategy.reduce(tensor, reduce_op="sum"))

    @staticmethod
    def _totals(terms: LossTerms) -> Tensor:
        return torch.stack((terms.loss_sum.detach().to(torch.float64),
                            terms.token_count.detach().to(torch.float64)))

    def setup(self, stage: str) -> None:
        self._ema_ready = False

    def _prepare_ema(self) -> None:
        """Place and synchronize EMA once after distributed model setup."""
        if self.ema is not None and not self._ema_ready:
            self.ema.to(device=self.device)
            if self.trainer.world_size > 1:
                self.ema.synchronize()
            self._ema_ready = True

    def on_fit_start(self) -> None:
        from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy
        if not isinstance(self.trainer.strategy, (DDPStrategy, SingleDeviceStrategy)):
            raise ValueError("%s training supports single-device and ordinary DDP"
                             % self.training_name)
        if self.trainer.precision not in ("32-true", "bf16-mixed"):
            raise ValueError("%s training supports FP32 and BF16 mixed precision"
                             % self.training_name)
        self._prepare_ema()
        self._validation_in_progress = False
        self._window_count = torch.zeros((), device=self.device, dtype=torch.float64)
        # Mid-epoch resume does not necessarily call on_train_epoch_start.
        self._train_totals = (torch.zeros(2, device=self.device, dtype=torch.float64)
                              if self._resume_train_totals is None else
                              self._resume_train_totals.to(self.device) / self.trainer.world_size)
        self._resume_train_totals = None

    def on_train_epoch_start(self) -> None:
        self._ensure_train_metric_epoch()

    def _ensure_train_metric_epoch(self) -> None:
        # Lightning can skip the epoch-start hook on resume, including a save
        # after the final batch. Keep mid-epoch sums, but never mix two epochs.
        if self._train_totals_epoch != self.current_epoch:
            self._train_totals = torch.zeros(2, device=self.device, dtype=torch.float64)
            self._train_totals_epoch = self.current_epoch

    def training_step(self, batch: TrainingBatch, batch_idx: int) -> Tensor:
        del batch_idx
        self._ensure_train_metric_epoch()
        terms = self._terms(batch)
        totals = self._totals(terms)
        # These accumulators are initialized by on_fit_start before any batch.
        self._window_count = cast(Tensor, self._window_count)
        self._train_totals = cast(Tensor, self._train_totals)
        self._window_count += totals[1]
        self._train_totals += totals
        # Reduce sums/counts, not an average of possibly unequal local means.
        logged = self._sum_ranks(totals.clone())
        denominator = torch.where(logged[1] > 0, logged[1], 1.)
        self.log("trainer/loss", logged[0] / denominator,
                 on_step=True, on_epoch=False, sync_dist=False)
        return terms.loss_sum

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        count = self._sum_ranks(cast(Tensor, self._window_count).clone())
        if not bool(torch.isfinite(count) & (count > 0)):
            raise ValueError("an optimizer window must contain positive finite token count")
        scale = (self.trainer.world_size * self.trainer.accumulate_grad_batches) / count
        scales: dict[torch.dtype, Tensor] = {}
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    dtype = parameter.grad.dtype
                    if dtype not in scales:
                        scales[dtype] = scale.to(dtype)
                    parameter.grad.mul_(scales[dtype])
        cast(Tensor, self._window_count).zero_()

    def _metric_values(self, mean: Tensor, token_count: Tensor) -> dict[str, Tensor]:
        return {"nll": mean, "ppl": mean.exp(), "token_count": token_count}

    def _publish_metrics(self, totals: Tensor, prefix: str,
                         metric_totals: Mapping[str, Tensor] | None = None) -> dict[str, Tensor]:
        names = sorted(metric_totals or {})
        if names:
            totals = torch.cat((totals, torch.stack([
                cast(Mapping[str, Tensor], metric_totals)[name] for name in names])))
        totals = self._sum_ranks(totals.clone())
        if not bool(totals[1] > 0):
            raise ValueError("%s contains no scored tokens" % prefix)
        values = self._metric_values(totals[0] / totals[1], totals[1])
        values.update({name: totals[index + 2] / totals[1]
                       for index, name in enumerate(names)})
        for name, value in values.items():
            self.log(prefix + "/" + name, value, on_step=False, on_epoch=True,
                     sync_dist=False)
        return {name: value.detach().clone() for name, value in values.items()}

    def on_train_epoch_end(self) -> None:
        self.last_train_metrics = self._publish_metrics(cast(Tensor, self._train_totals), "train")

    def on_validation_epoch_start(self) -> None:
        self._validation_in_progress = True
        self._valid_totals = torch.zeros(2, device=self.device, dtype=torch.float64)
        self._valid_metric_totals = {}
        self._prepare_ema()
        if self.ema is not None:
            context = self.ema.average_parameters(self.model)
            context.__enter__()
            self._ema_selection = context

    def _restore_validation_weights(self) -> None:
        if self._ema_selection is not None:
            context, self._ema_selection = self._ema_selection, None
            context.__exit__(None, None, None)

    def validation_step(self, batch: TrainingBatch, batch_idx: int) -> Tensor:
        del batch_idx
        try:
            terms = self._validation_terms(batch)
            self._valid_totals = cast(Tensor, self._valid_totals)
            self._valid_totals += self._totals(terms)
            for name, value in (terms.metrics or {}).items():
                self._valid_metric_totals[name] = (
                    self._valid_metric_totals.get(name, 0.) + value.detach().to(torch.float64))
            return terms.loss_sum.detach()
        except BaseException:
            self._restore_validation_weights()
            raise

    def on_validation_epoch_end(self) -> None:
        try:
            self.last_validation_metrics = self._publish_metrics(
                cast(Tensor, self._valid_totals), "val", self._valid_metric_totals)
            self._validation_in_progress = False
        finally:
            self._restore_validation_weights()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(self.model.parameters(), **asdict(self.optimizer_config))
        # Update EMA once per completed optimizer step.
        if self.ema is not None:
            optimizer.register_step_post_hook(
                lambda *_: cast(ExponentialMovingAverage, self.ema).update(self.model))
        scheduler = build_lr_scheduler(optimizer, self.schedule_config)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def _contract(self) -> TrainingContract:
        return {"blocked_token_ids": self.blocked_token_ids,
                "optimizer": asdict(self.optimizer_config),
                "schedule": asdict(self.schedule_config),
                "model": asdict(self.model.config),
                "ema_decay": self.ema.decay if self.ema is not None else None}

    def on_save_checkpoint(self, checkpoint: Checkpoint) -> None:
        if self._ema_selection is not None:
            raise RuntimeError("cannot checkpoint temporarily selected EMA weights")
        if self._validation_in_progress:
            raise RuntimeError("save %s checkpoints after completed validation, not during a failed validation"
                               % self.training_name)
        if self._window_count is not None:
            if bool(self._window_count != 0):
                raise RuntimeError("save %s checkpoints at optimizer-update boundaries"
                                   % self.training_name)
            loader = self.trainer.train_dataloader
            if getattr(loader, "has_pending_batch", False):
                raise RuntimeError("save %s checkpoints after the completed-batch callback"
                                   % self.training_name)
        state: TokenCheckpointState = {
            "version": 1, "contract": self._contract(),
            "ema": self.ema.state_dict() if self.ema is not None else None,
            "train_totals_epoch": self._train_totals_epoch,
            "train_totals": (None if self._train_totals is None else
                             self._sum_ranks(self._train_totals.clone()))}
        checkpoint[self.checkpoint_key] = state

    def on_load_checkpoint(self, checkpoint: Checkpoint) -> None:
        state = cast(TokenCheckpointState, checkpoint[self.checkpoint_key])
        if state["version"] != 1 or state["contract"] != self._contract():
            raise ValueError("%s resume contract differs; initialization is not resume"
                             % self.training_name)
        if self.ema is not None:
            self.ema.load_state_dict(cast(Mapping[str, Any], state["ema"]))
        self._ema_ready = False
        self._resume_train_totals = state["train_totals"]
        self._train_totals_epoch = state["train_totals_epoch"]


class _TokenLifecycle(Callback):
    """Restore live weights before later callbacks can save on an exception."""

    def on_exception(self, trainer: Trainer, pl_module: LightningModule,
                     exception: BaseException) -> None:
        del trainer, exception
        cast("TokenTrainingModule[NativeModel]", pl_module)._restore_validation_weights()


def fit_token_model(module: TokenTrainingModule[Any], datamodule: LightningDataModule, *,
                    trainer: Trainer, ckpt_path: Pathish | None = None) -> Trainer:
    """Fit with data progress before checkpoints and live-weight cleanup."""
    from data.loading import TrainingDataProgress
    callback_trainer = cast(TrainerCallbacks, trainer)
    callback_trainer.callbacks = [callback for callback in callback_trainer.callbacks
                         if not isinstance(callback, (TrainingDataProgress, _TokenLifecycle))]
    callback_trainer.callbacks[:0] = [_TokenLifecycle(), TrainingDataProgress()]
    try:
        trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)
    finally:
        module._restore_validation_weights()
    return trainer


def validate_token_model(module: TokenTrainingModule[Any], datamodule: LightningDataModule, *,
                         trainer: Trainer) -> ValidationResults:
    """Validate with temporary EMA selection and exception cleanup."""
    callback_trainer = cast(TrainerCallbacks, trainer)
    callback_trainer.callbacks = [callback for callback in callback_trainer.callbacks
                         if not isinstance(callback, _TokenLifecycle)]
    callback_trainer.callbacks.insert(0, _TokenLifecycle())
    try:
        return trainer.validate(module, datamodule=datamodule)
    finally:
        module._restore_validation_weights()
