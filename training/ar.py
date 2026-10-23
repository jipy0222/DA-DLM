"""AR training with caller-supplied models, target weights and Lightning configuration."""

from __future__ import annotations

from typing import Optional, Sequence

from lightning.pytorch import LightningDataModule, Trainer

from models import ARModel
from objectives import LossTerms, ar_objective, prepare_ar_batch
from training._types import Pathish, TrainingBatch, ValidationResults
from training.lr_schedules import LRScheduleConfig
from training.token import (
    AdamWConfig,
    TokenTrainingModule,
    fit_token_model,
    validate_token_model,
)


class ARTrainingModule(TokenTrainingModule[ARModel]):
    """Token-weighted AR training with optimizer-step EMA and LR schedules.

    PAD targets are never scored. Task formatting supplies optional loss_mask
    weights. For summary targets, exclude source/summary BOS positions and leave
    summary EOS unblocked in the output vocabulary.
    """

    checkpoint_key = "ar_pretraining"
    training_name = "AR"

    def __init__(self, model: ARModel, *, blocked_token_ids: Sequence[int],
                 optimizer: AdamWConfig = AdamWConfig(),
                 schedule: LRScheduleConfig = LRScheduleConfig(),
                 ema_decay: Optional[float] = 0.9999) -> None:
        super().__init__(model, blocked_token_ids=blocked_token_ids,
                         optimizer=optimizer, schedule=schedule, ema_decay=ema_decay)

    def _terms(self, batch: TrainingBatch) -> LossTerms:
        inputs, targets, mask = prepare_ar_batch(
            batch["input_ids"], batch.get("loss_mask"), batch.get("sample_weight"))
        mask = mask.masked_fill(targets.eq(self.model.config.pad_token_id), 0)
        return ar_objective(self.model(inputs).logits, targets, mask,
                            blocked_token_ids=self.blocked_token_ids)


def fit_ar(module: ARTrainingModule, datamodule: LightningDataModule, *, trainer: Trainer,
           ckpt_path: Pathish | None = None) -> Trainer:
    """Run AR through the shared token training lifecycle."""
    return fit_token_model(module, datamodule, trainer=trainer, ckpt_path=ckpt_path)


def validate_ar(module: ARTrainingModule, datamodule: LightningDataModule, *,
                trainer: Trainer) -> ValidationResults:
    """Validate with temporary EMA selection, including exception cleanup."""
    return validate_token_model(module, datamodule, trainer=trainer)
