"""Block diffusion training with noisy targets and clean block-causal context."""

from __future__ import annotations

from typing import Sequence

from lightning.pytorch import LightningDataModule, Trainer
from torch import Tensor

from models import BD3LMModel
from models.backbone.types import TokenModelOutput
from objectives.noise import sample_block_time
from training._types import Pathish, TrainingBatch, TrainingContract, ValidationResults
from training.diffusion import DiffusionTrainingConfig as BD3LMTrainingConfig
from training.diffusion import DiffusionTrainingModule
from training.lr_schedules import LRScheduleConfig
from training.token import AdamWConfig, fit_token_model, validate_token_model


class BD3LMTrainingModule(DiffusionTrainingModule[BD3LMModel]):
    """Reconstruct target blocks with their preceding clean context.

    ``input_ids`` contains the complete clean sequence. ``source_length`` is
    its fixed prefix width (zero for pretraining); the suffix is the target,
    including summary BOS when present. Loss masks have the full input width.
    Summary formatting excludes source/BOS/PAD and leaves summary EOS scored;
    callers must also leave summary EOS allowed in the output vocabulary.
    With ignore_bos=True, bos_token_id identifies the first target token:
    ordinary BOS for pretraining or summary BOS after the source prefix.
    That token is checked and excluded from every phase's denominator.

    The model receives (noisy target, clean input_ids). Its block layout, not
    concatenation alone, prevents attention to the current/future clean target.
    Configure its physical capacity with BD3LMConfig.with_lengths before model
    construction. Source-boundary alignment is owned by that model; neither
    the loader nor the training objective adds target PAD or relocates EOS.
    """

    checkpoint_key = "bd3lm_training"
    training_name = "BD3LM"

    def __init__(self, model: BD3LMModel, *, mask_token_id: int, blocked_token_ids: Sequence[int],
                 bos_token_id: int | None = None,
                 source_length: int = 0,
                 diffusion: BD3LMTrainingConfig = BD3LMTrainingConfig(),
                 optimizer: AdamWConfig = AdamWConfig(),
                 schedule: LRScheduleConfig = LRScheduleConfig(), ema_decay: float | None = .9999) -> None:
        if type(source_length) is not int or source_length < 0:
            raise ValueError("source_length must be a nonnegative slot count")
        super().__init__(model, mask_token_id=mask_token_id,
                         bos_token_id=bos_token_id,
                         blocked_token_ids=blocked_token_ids, diffusion=diffusion,
                         optimizer=optimizer, schedule=schedule, ema_decay=ema_decay)
        self.source_length = source_length

    def _target_data(self, batch: TrainingBatch) -> tuple[Tensor, Tensor | None]:
        ids = batch["input_ids"]
        if ids.ndim != 2 or ids.shape[1] <= self.source_length:
            raise ValueError("input_ids must contain a nonempty target after source_length")
        mask = batch.get("loss_mask")
        return (ids[:, self.source_length:],
                None if mask is None else mask[:, self.source_length:])

    def _sample_times(self, input_ids: Tensor, *, t_min: float, t_max: float,
                      antithetic: bool) -> Tensor:
        return sample_block_time(
            input_ids.shape[0], input_ids.shape[1], self.model.config.block_size,
            input_ids.device, antithetic=antithetic,
            t_min=t_min, t_max=t_max)

    @property
    def _full_mask(self) -> bool:
        return self.diffusion_config.full_mask or self.model.config.block_size == 1

    def _predict_logits(self, noisy_ids: Tensor, batch: TrainingBatch) -> Tensor:
        output: TokenModelOutput = self.model(noisy_ids, batch["input_ids"])
        return output.logits

    def _contract(self) -> TrainingContract:
        return {**super()._contract(), "source_length": self.source_length}


def fit_bd3lm(module: BD3LMTrainingModule, datamodule: LightningDataModule, *,
              trainer: Trainer, ckpt_path: Pathish | None = None) -> Trainer:
    """Fit prepared data with completed-batch tracking and optimizer-step EMA."""
    return fit_token_model(module, datamodule, trainer=trainer, ckpt_path=ckpt_path)


def validate_bd3lm(module: BD3LMTrainingModule, datamodule: LightningDataModule, *,
                   trainer: Trainer) -> ValidationResults:
    """Validate using the full two-stream objective, repeated noise draws and EMA."""
    return validate_token_model(module, datamodule, trainer=trainer)
