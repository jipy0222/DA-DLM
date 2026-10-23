"""MDLM reconstruction training on clean sequences and explicit target weights."""

from __future__ import annotations

from typing import Sequence

from lightning.pytorch import LightningDataModule, Trainer
from torch import Tensor

from models import MDLMModel
from models.backbone.types import TokenModelOutput
from training._types import Pathish, TrainingBatch, ValidationResults
from training.diffusion import DiffusionTrainingConfig as MDLMTrainingConfig
from training.diffusion import DiffusionTrainingModule
from training.lr_schedules import LRScheduleConfig
from training.token import AdamWConfig, fit_token_model, validate_token_model


class MDLMTrainingModule(DiffusionTrainingModule[MDLMModel]):
    """Train a zero-time-conditioned MDLM using the shared token training loop.

    Clean input_ids remain unshifted. PAD is never corrupted or scored. An
    optional loss_mask selects prediction targets; other positions stay visible
    to bidirectional attention. The output vocabulary policy is caller-supplied.

    Summary rows keep source, summary BOS and PAD clean through loss_mask.
    Summary text and summary EOS are reconstruction targets; the caller must
    leave summary EOS unblocked.

    By default every row must start with bos_token_id. That position stays
    clean and is excluded from the denominator in training and all validation.
    For summary inputs starting with source text, set ignore_bos=False and
    let loss_mask protect source/summary BOS.
    """

    checkpoint_key = "mdlm_training"
    training_name = "MDLM"

    def __init__(self, model: MDLMModel, *, mask_token_id: int, blocked_token_ids: Sequence[int],
                 bos_token_id: int | None = None,
                 diffusion: MDLMTrainingConfig = MDLMTrainingConfig(),
                 optimizer: AdamWConfig = AdamWConfig(),
                 schedule: LRScheduleConfig = LRScheduleConfig(), ema_decay: float | None = .9999) -> None:
        super().__init__(model, mask_token_id=mask_token_id,
                         bos_token_id=bos_token_id,
                         blocked_token_ids=blocked_token_ids, diffusion=diffusion,
                         optimizer=optimizer, schedule=schedule, ema_decay=ema_decay)

    def _predict_logits(self, noisy_ids: Tensor, batch: TrainingBatch) -> Tensor:
        output: TokenModelOutput = self.model(noisy_ids)
        return output.logits


def fit_mdlm(module: MDLMTrainingModule, datamodule: LightningDataModule, *,
              trainer: Trainer, ckpt_path: Pathish | None = None) -> Trainer:
    """Run MDLM through the shared token training lifecycle."""
    return fit_token_model(module, datamodule, trainer=trainer, ckpt_path=ckpt_path)


def validate_mdlm(module: MDLMTrainingModule, datamodule: LightningDataModule, *,
                   trainer: Trainer) -> ValidationResults:
    """Estimate the diffusion objective with fresh masking and temporary EMA."""
    return validate_token_model(module, datamodule, trainer=trainer)
