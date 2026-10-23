"""Shared masked-diffusion training for token reconstruction models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Generic, Sequence, TypeVar, Union, cast

import torch
from torch import Tensor

from models import BD3LMModel, MDLMModel
from models.backbone.config import ConditioningMode
from objectives import (
    LossTerms,
    MaskingSchedule,
    mask_tokens,
    mdlm_objective,
    prepare_mdlm_batch,
    sample_mdlm_time,
)
from training._types import DiffusionDraw, TrainingBatch, TrainingContract
from training.lr_schedules import LRScheduleConfig
from training.token import AdamWConfig, TokenTrainingModule

_Inputs = TypeVar("_Inputs")
_DiffusionModel = TypeVar("_DiffusionModel", bound=Union[MDLMModel, BD3LMModel])


@dataclass(frozen=True)
class DiffusionTrainingConfig:
    """Masking and validation settings, independent of optimizer LR schedules.

    Training uses ``noise`` and [t_min, t_max); validation uses loglinear on
    (0, 1), with independent uniform times and ``num_eval_repeats`` Monte Carlo
    draws per batch. ``antithetic_sampling`` controls training only. A full-mask
    objective instead masks every eligible token, uses unit loss weights and
    evaluates once without sampling noise.
    ``ignore_bos`` checks and excludes the first target token in every phase.
    """

    noise: MaskingSchedule = MaskingSchedule()
    t_min: float = 1e-3
    t_max: float = 1.
    antithetic_sampling: bool = True
    ignore_bos: bool = True
    full_mask: bool = False
    num_eval_repeats: int = 1

    def __post_init__(self) -> None:
        if not 0 < self.t_min < self.t_max <= 1:
            raise ValueError("training times must satisfy 0 < t_min < t_max <= 1")
        if type(self.num_eval_repeats) is not int or self.num_eval_repeats < 1:
            raise ValueError("num_eval_repeats must be a positive integer")


def diffusion_validation_terms(draw: DiffusionDraw[_Inputs], inputs: _Inputs, *,
                               full_mask: bool, num_eval_repeats: int) -> LossTerms:
    """Average independent validation draws, counting prediction tokens once.

    Full masking is deterministic in eval mode and needs only one forward.
    Otherwise each draw uses independent loglinear noise over the full range.
    """
    repeats = 1 if full_mask else num_eval_repeats
    noise = MaskingSchedule()
    total: Tensor | None = None
    metrics: dict[str, Tensor] = {}
    for _ in range(repeats):
        terms = draw(inputs, noise=noise, t_min=0., t_max=1.,
                     antithetic=False)
        total = terms.loss_sum if total is None else total + terms.loss_sum
        for name, value in (terms.metrics or {}).items():
            metrics[name] = metrics.get(name, 0.) + value.detach()
    # DiffusionTrainingConfig guarantees a positive repeat count.
    return LossTerms(cast(Tensor, total) / repeats, terms.token_count,  # ty: ignore[possibly-unresolved-reference]
                     {name: value / repeats for name, value in metrics.items()} or None)


class DiffusionTrainingModule(TokenTrainingModule[_DiffusionModel], Generic[_DiffusionModel]):
    """Token-weighted masked diffusion with overridable targets and logits."""

    def __init__(self, model: _DiffusionModel, *, mask_token_id: int,
                 blocked_token_ids: Sequence[int],
                 bos_token_id: int | None = None,
                 diffusion: DiffusionTrainingConfig = DiffusionTrainingConfig(),
                 optimizer: AdamWConfig = AdamWConfig(),
                 schedule: LRScheduleConfig = LRScheduleConfig(), ema_decay: float | None = .9999) -> None:
        if (type(mask_token_id) is not int or
                not 0 <= mask_token_id < model.config.vocab_size or
                mask_token_id == model.config.pad_token_id):
            raise ValueError("MASK must be a vocabulary ID distinct from PAD")
        if model.config.conditioning is not ConditioningMode.ZERO_TIME_COMPAT:
            raise ValueError("this diffusion training objective uses ZERO_TIME_COMPAT conditioning")
        if diffusion.ignore_bos and bos_token_id is None:
            raise ValueError("ignore_bos=True requires bos_token_id")
        if bos_token_id is not None and (
                type(bos_token_id) is not int or not 0 <= bos_token_id < model.config.vocab_size or
                bos_token_id in (mask_token_id, model.config.pad_token_id)):
            raise ValueError("BOS must be a vocabulary ID distinct from MASK and PAD")
        blocked_token_ids = tuple(blocked_token_ids)
        if mask_token_id not in blocked_token_ids or model.config.pad_token_id not in blocked_token_ids:
            raise ValueError("block MASK and PAD in the reconstruction vocabulary")
        super().__init__(model, blocked_token_ids=blocked_token_ids, optimizer=optimizer,
                         schedule=schedule, ema_decay=ema_decay)
        self.mask_token_id = mask_token_id
        self.bos_token_id = bos_token_id
        self.diffusion_config = diffusion

    def _target_data(self, batch: TrainingBatch) -> tuple[Tensor, Tensor | None]:
        return batch["input_ids"], batch.get("loss_mask")

    def _sample_times(self, input_ids: Tensor, *, t_min: float, t_max: float,
                      antithetic: bool) -> Tensor:
        return sample_mdlm_time(
            input_ids.shape[0], input_ids.device,
            antithetic=antithetic, t_min=t_min, t_max=t_max)

    def _draw_noise(self, input_ids: Tensor, *, t_min: float, t_max: float,
                    antithetic: bool) -> tuple[Tensor, Tensor]:
        times = self._sample_times(input_ids, t_min=t_min, t_max=t_max, antithetic=antithetic)
        uniform = torch.rand(input_ids.shape, device=input_ids.device)
        return times, uniform

    def _predict_logits(self, noisy_ids: Tensor, batch: TrainingBatch) -> Tensor:
        raise NotImplementedError

    @property
    def _full_mask(self) -> bool:
        return self.diffusion_config.full_mask

    def _diffusion_terms(self, batch: TrainingBatch, *, noise: MaskingSchedule,
                         t_min: float, t_max: float, antithetic: bool) -> LossTerms:
        ids, loss_mask = self._target_data(batch)
        eligible, weights = prepare_mdlm_batch(
            ids, pad_token_id=self.model.config.pad_token_id,
            loss_mask=loss_mask, sample_weight=batch.get("sample_weight"),
            ignore_bos=self.diffusion_config.ignore_bos, bos_token_id=self.bos_token_id)
        if self._full_mask:
            noisy_ids = ids.masked_fill(eligible, self.mask_token_id)
            loss_weight = weights.new_ones(())
        else:
            times, uniform = self._draw_noise(ids, t_min=t_min, t_max=t_max, antithetic=antithetic)
            loss_weight, probability = noise(times)
            noisy_ids = mask_tokens(ids, probability, mask_token_id=self.mask_token_id,
                                    corruption_mask=eligible, uniform=uniform)
        return mdlm_objective(self._predict_logits(noisy_ids, batch), ids, noisy_ids,
                              loss_weight, weights, mask_token_id=self.mask_token_id,
                              blocked_token_ids=self.blocked_token_ids)

    def _terms(self, batch: TrainingBatch) -> LossTerms:
        config = self.diffusion_config
        return self._diffusion_terms(batch, noise=config.noise, t_min=config.t_min,
                                     t_max=config.t_max,
                                     antithetic=config.antithetic_sampling)

    def _validation_terms(self, batch: TrainingBatch) -> LossTerms:
        return diffusion_validation_terms(
            self._diffusion_terms, batch, full_mask=self._full_mask,
            num_eval_repeats=self.diffusion_config.num_eval_repeats)

    def _contract(self) -> TrainingContract:
        return {**super()._contract(), "mask_token_id": self.mask_token_id,
                "bos_token_id": self.bos_token_id,
                "diffusion": asdict(self.diffusion_config)}
