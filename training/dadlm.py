"""DADLM block objectives with shared token-weighted training and validation."""

from __future__ import annotations

from dataclasses import asdict
from functools import partial
from typing import cast

import torch
from lightning.pytorch import LightningDataModule, Trainer
from torch import Tensor

from data.tokens import NativeTokenSchema
from graph.runtime import GraphBackend, validate_backend
from models.backbone.config import ConditioningMode, TransitionNormalization
from models.backbone.types import DADModelOutput
from models.dadlm import DADLMModel
from objectives.assignment import (
    aligned_assignment,
    initial_assignment,
    validate_assignment,
)
from objectives.coherence import coherence_gap
from objectives.dadlm import build_emissions, dadlm_objective
from objectives.dag import DAGLayout, DAGTargets, build_dag, prepare_dag_targets
from objectives.mdlm import mask_tokens
from objectives.noise import MaskingSchedule, sample_block_time
from objectives.types import LossTerms
from training._types import Pathish, TrainingBatch, TrainingContract, ValidationResults
from training.diffusion import DiffusionTrainingConfig, diffusion_validation_terms
from training.graph_runtime import prepare_graph_runtime
from training.lr_schedules import LRScheduleConfig
from training.token import (
    AdamWConfig,
    TokenTrainingModule,
    fit_token_model,
    validate_token_model,
)


class DADLMTrainingModule(TokenTrainingModule[DADLMModel]):
    """Train fixed target blocks by marginalizing all compatible DAG paths.

    ``input_ids`` is the clean source+target sequence. ``source_length=0``
    selects pretraining token roles; a nonempty source selects summary roles.
    Summary data retain their existing source/BOS/EOS/PAD format. The optional
    loss mask must match those task roles, not arbitrary token-level weights.

    Configure model capacity using the target width rounded up to block_size,
    in both current_length and context_length. Physical DAG endpoints remain
    fixed even for a partially filled target block.
    """

    checkpoint_key = "dadlm_training"
    training_name = "DADLM"

    def __init__(self, model: DADLMModel, *, tokens: NativeTokenSchema,
                 assignment: str | None = None, source_length: int = 0,
                 diffusion: DiffusionTrainingConfig = DiffusionTrainingConfig(),
                 optimizer: AdamWConfig = AdamWConfig(),
                 schedule: LRScheduleConfig = LRScheduleConfig(), ema_decay: float | None = .9999,
                 graph_backend: GraphBackend = "auto", warmup_before_training: bool = True,
                 monitor_coherence: bool = False) -> None:
        config = model.config
        if tokens.vocab_size != config.vocab_size or tokens.pad_token_id != config.pad_token_id:
            raise ValueError("token schema must match the model vocabulary and PAD")
        if type(source_length) is not int or source_length < 0:
            raise ValueError("source_length must be a nonnegative slot count")
        if config.conditioning is not ConditioningMode.ZERO_TIME_COMPAT:
            raise ValueError("DADLM training uses ZERO_TIME_COMPAT conditioning")
        if config.nodes_per_block - 2 < config.block_size:
            raise ValueError("DAG content capacity must fit a full target block")
        if not diffusion.ignore_bos:
            raise ValueError("DADLM task inputs require a fixed, excluded initial BOS")
        if assignment is None:
            assignment = "canonical" if config.grouped else "uniform"
        self.assignment = validate_assignment(assignment, config.grouped)
        self.graph_backend = validate_backend(graph_backend)
        if type(warmup_before_training) is not bool:
            raise TypeError("warmup_before_training must be boolean")
        self.warmup_before_training = warmup_before_training
        self._graph_ready = False
        self.tokens = tokens
        self.source_length = source_length
        self.bos_token_id = tokens.summary_bos if source_length else tokens.bos_token_id
        self.terminal_token_id = tokens.summary_eos if source_length else None
        blocked = tokens.blocked_token_ids
        if source_length:
            blocked = tuple(index for index in blocked if index != tokens.summary_eos)
        super().__init__(model, blocked_token_ids=blocked, optimizer=optimizer,
                         schedule=schedule, ema_decay=ema_decay)
        self.diffusion_config = diffusion
        self.monitor_coherence = monitor_coherence

    def _prepare(self, batch: TrainingBatch) -> DAGTargets:
        targets = prepare_dag_targets(
            batch["input_ids"], block_size=self.model.config.block_size,
            pad_token_id=self.tokens.pad_token_id, bos_token_id=self.bos_token_id,
            source_length=self.source_length, loss_mask=batch.get("loss_mask"),
            sample_weight=batch.get("sample_weight"))
        ordinary = targets.tokens.ne(self.tokens.pad_token_id)
        ordinary[:, 0] &= ~targets.fixed_bos
        blocked = targets.tokens.new_tensor(self.blocked_token_ids)
        if bool((ordinary & torch.isin(targets.tokens, blocked)).any()):
            raise ValueError("prediction targets contain a forbidden output token")
        if self.terminal_token_id is not None:
            suffix = targets.context_ids[:, self.source_length:]
            lengths = suffix.ne(self.tokens.pad_token_id).sum(-1)
            last = suffix.gather(1, (lengths - 1)[:, None]).squeeze(1)
            if not bool(last.eq(self.terminal_token_id).all() & suffix.eq(self.terminal_token_id).sum(-1).eq(1).all()):
                raise ValueError("summary targets must end in exactly one summary EOS")
        return targets

    def _draw_noise(self, input_ids: Tensor, *, t_min: float, t_max: float,
                    antithetic: bool) -> tuple[Tensor, Tensor]:
        times = sample_block_time(
            input_ids.shape[0], input_ids.shape[1], self.model.config.block_size,
            input_ids.device, t_min=t_min, t_max=t_max, antithetic=antithetic)
        return times, torch.rand(input_ids.shape, device=input_ids.device)

    def _layout(self, targets: DAGTargets, noisy: Tensor, placements: Tensor) -> DAGLayout:
        schema, config = self.tokens, self.model.config
        return build_dag(targets, noisy, placements,
                         dag_up_ratio=config.dag_up_ratio, grouped=config.grouped,
                         mask_token_id=schema.mask_token_id,
                         pad_token_id=schema.pad_token_id,
                         block_bos=schema.block_bos, block_eos=schema.block_eos,
                         terminal_token_id=self.terminal_token_id)

    def _predict(self, targets: DAGTargets, layout: DAGLayout) -> DADModelOutput:
        return cast(DADModelOutput, self.model(layout.ids.reshape(targets.batch_size, -1),
                          targets.context_ids, layout.edge_valid))

    def _emissions(self, output: DADModelOutput, targets: DAGTargets, layout: DAGLayout,
                   partial_mask: Tensor | None = None) -> Tensor:
        logits = output.logits.reshape(-1, self.model.config.nodes_per_block,
                                       self.tokens.vocab_size)
        return build_emissions(
            logits, targets.expanded(self.tokens.block_bos, self.tokens.block_eos,
                                     self.tokens.pad_token_id),
            targets.lengths + 2, layout.forced_nodes,
            layout.ids.ne(self.tokens.pad_token_id),
            blocked_token_ids=self.blocked_token_ids, partial_mask=partial_mask,
            backend=self.graph_backend)

    def _placements(self, targets: DAGTargets, noisy: Tensor) -> Tensor:
        config, schema = self.model.config, self.tokens
        if not self.diffusion_config.full_mask and self.assignment in ("canonical", "uniform", "random"):
            return initial_assignment(targets.lengths, block_size=config.block_size,
                                      dag_up_ratio=config.dag_up_ratio,
                                      strategy=self.assignment, fixed_bos=targets.fixed_bos)
        placements = torch.full_like(targets.tokens, -1)
        placements[targets.fixed_bos, 0] = 1
        # Only fixed BOS is visible under full masking; no other placement
        # can affect the input graph or its emission constraints.
        if self.diffusion_config.full_mask:
            return placements
        active = targets.counts > 0
        if not bool(active.any()):
            return placements
        # A no-grad prepass must not cache detached low-precision weights for
        # the subsequent differentiable forward in the same autocast scope.
        device_type = targets.tokens.device.type
        with torch.no_grad(), torch.autocast(
                device_type, enabled=torch.is_autocast_enabled(device_type),
                cache_enabled=False):
            preliminary = torch.full_like(noisy, schema.mask_token_id)
            preliminary = preliminary.masked_fill(targets.tokens.eq(schema.pad_token_id), schema.pad_token_id)
            preliminary[targets.fixed_bos, 0] = self.bos_token_id
            layout = self._layout(targets, preliminary, placements)
            output = self._predict(targets, layout)
            partial = None
            if self.assignment.startswith("partial_"):
                partial = torch.zeros_like(layout.forced_nodes, dtype=torch.bool)
                partial[:, 1:-1] = noisy.eq(schema.mask_token_id)
            emissions = self._emissions(output, targets, layout, partial)
            placements[active] = aligned_assignment(
                emissions[active], output.links.scores[active], targets.lengths[active],
                strategy=self.assignment, backend=self.graph_backend)
        return placements

    def _one_draw(self, targets: DAGTargets, *, noise: MaskingSchedule, t_min: float,
                  t_max: float, antithetic: bool, monitor_coherence: bool = False) -> LossTerms:
        ids = targets.tokens.reshape(targets.batch_size, -1)
        eligible = ids.ne(self.tokens.pad_token_id)
        eligible[:, 0] = False
        if self.diffusion_config.full_mask:
            noisy = ids.masked_fill(eligible, self.tokens.mask_token_id)
            weights = torch.ones_like(targets.counts)
        else:
            times, uniforms = self._draw_noise(ids, t_min=t_min, t_max=t_max, antithetic=antithetic)
            weights, probability = noise(times)
            noisy = mask_tokens(ids, probability, mask_token_id=self.tokens.mask_token_id,
                                corruption_mask=eligible, uniform=uniforms)
            weights = weights.reshape_as(targets.tokens)[:, 0]
        noisy = noisy.reshape_as(targets.tokens)
        placements = self._placements(targets, noisy)
        layout = self._layout(targets, noisy, placements)
        output = self._predict(targets, layout)
        emissions = self._emissions(output, targets, layout)
        terms = dadlm_objective(
            emissions, output.links.scores, targets.lengths + 2, targets.counts,
            loss_weights=weights,
            sample_weights=targets.sample_weights,
            crf=self.model.config.transition_normalization is TransitionNormalization.RAW_POTENTIAL,
            backend=self.graph_backend)
        if not monitor_coherence:
            return terms
        active = (targets.counts > 0) & (targets.sample_weights > 0)
        gap_sum = terms.loss_sum.detach().new_zeros(())
        if bool(active.any()):
            gaps = coherence_gap(
                emissions[active], output.links.scores[active], targets.lengths[active] + 2,
                layout.forced_nodes[active], layout.ids[active].ne(self.tokens.pad_token_id),
                backend=self.graph_backend)
            gap_sum = (gaps * targets.sample_weights[active]).sum()
        return LossTerms(terms.loss_sum, terms.token_count, {"coherence_gap": gap_sum})

    def _terms(self, batch: TrainingBatch) -> LossTerms:
        config = self.diffusion_config
        return self._one_draw(self._prepare(batch), noise=config.noise,
                              t_min=config.t_min, t_max=config.t_max,
                              antithetic=config.antithetic_sampling)

    def _validation_terms(self, batch: TrainingBatch) -> LossTerms:
        return diffusion_validation_terms(
            partial(self._one_draw, monitor_coherence=self.monitor_coherence), self._prepare(batch),
            full_mask=self.diffusion_config.full_mask,
            num_eval_repeats=self.diffusion_config.num_eval_repeats)

    def _metric_values(self, mean: Tensor, token_count: Tensor) -> dict[str, Tensor]:
        return {"loss": mean, "token_count": token_count}

    def _prepare_graph(self) -> None:
        if not self._graph_ready:
            if self.graph_backend == "cuda" and self.device.type != "cuda":
                raise ValueError("graph_backend='cuda' requires a CUDA model")
            backend = self.graph_backend if self.device.type == "cuda" else "pytorch"
            prepare_graph_runtime(backend=backend, warmup_before_training=self.warmup_before_training)
            self._graph_ready = True

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self._prepare_graph()

    def on_validation_epoch_start(self) -> None:
        self._prepare_graph()
        super().on_validation_epoch_start()

    def _contract(self) -> TrainingContract:
        return {**super()._contract(), "tokens": self.tokens.to_dict(),
                "source_length": self.source_length, "assignment": self.assignment,
                "diffusion": asdict(self.diffusion_config)}


def fit_dadlm(module: DADLMTrainingModule, datamodule: LightningDataModule, *,
              trainer: Trainer, ckpt_path: Pathish | None = None) -> Trainer:
    """Fit with global token normalization and completed-batch recovery."""
    return fit_token_model(module, datamodule, trainer=trainer, ckpt_path=ckpt_path)


def validate_dadlm(module: DADLMTrainingModule, datamodule: LightningDataModule, *,
                   trainer: Trainer) -> ValidationResults:
    """Evaluate the configured DAG assignment/objective with MC and EMA."""
    return validate_token_model(module, datamodule, trainer=trainer)
