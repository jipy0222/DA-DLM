"""Shared execution of prepared training and validation tasks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping, Sequence, Union, cast

from lightning.pytorch import Callback, Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from data.loading import TrainingDataModule
from data.pretrain_storage import TokenDataset
from data.tokenizer_io import check_vocabulary, load_tokenizer
from data.tokens import NativeTokenSchema
from training._types import (
    NativeModel,
    TrainingModule,
    ValidationResults,
)
from training.callbacks import (
    InitializationReport,
    _RunOutput,
    _UpdateCheckpoint,
    _UpdateValidationLoop,
)
from training.checkpoint import checkpoint_tokenizer
from training.configuration import plain_config, require_path, typed_training_options
from training.ema import ExponentialMovingAverage
from training.runner import build_trainer, build_validation_checkpoint
from training.token import (
    fit_token_model,
    validate_token_model,
)

if TYPE_CHECKING:
    from datasets import Dataset as ArrowDataset

    from training.checkpoint import NativeCheckpointLoad


PreparedDataset = Union["TokenDataset", "ArrowDataset"]


def require_single_preparation() -> None:
    """Keep offline data preparation outside distributed worker launches."""
    slurm_step_tasks = os.environ.get(
        "SLURM_STEP_NUM_TASKS", os.environ.get("SLURM_NTASKS", "1"))
    if (int(os.environ.get("WORLD_SIZE", "1")) != 1
            or ("SLURM_PROCID" in os.environ and int(slurm_step_tasks) != 1)):
        raise ValueError("run data preparation once, outside the distributed training launch")


def data_module(config: DictConfig, train: PreparedDataset | None,
                 valid: PreparedDataset) -> TrainingDataModule:
    runtime = config.runtime
    return TrainingDataModule(
        # Arrow and token stores both implement the map-style dataset protocol.
        train,
        valid, local_batch_size=runtime.local_batch_size,
        eval_batch_size=runtime.eval_batch_size, num_workers=runtime.loader_num_workers,
        seed=config.seed, pin_memory=runtime.pin_memory,
    )


def build_logger(config: DictConfig) -> Logger | Literal[False]:
    options = config.logging
    if options.backend == "none":
        return False
    from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
    classes = {"csv": CSVLogger, "tensorboard": TensorBoardLogger}
    if options.backend not in classes:
        raise ValueError("logging.backend must be tensorboard, csv or none")
    return cast(Logger, classes[options.backend](save_dir=options.save_dir, name=options.name))


def build_workflow_trainer(config: DictConfig, callbacks: list[Callback], *,
             validation_only: bool = False) -> Trainer:
    values = plain_config(config.runtime)
    for name in ("local_batch_size", "eval_batch_size", "loader_num_workers", "pin_memory"):
        values.pop(name)
    every = config.validation.every_updates
    if every is not None and (type(every) is not int or every < 1):
        raise ValueError("validation.every_updates must be positive or null")
    logger = build_logger(config)
    if not validation_only and logger is not False:
        callbacks = [*callbacks, LearningRateMonitor(logging_interval="step")]
    trainer = build_trainer(
        **values, logger=logger, callbacks=callbacks,
        default_root_dir=config.output_dir,
        enable_checkpointing=any(isinstance(callback, ModelCheckpoint) for callback in callbacks),
        num_sanity_val_steps=config.validation.num_sanity_steps if every is not None else 0,
        limit_val_batches=config.validation.limit_batches if (validation_only or every is not None) else 0,
        val_check_interval=config.runtime.accumulate_grad_batches * (every or 1),
        check_val_every_n_epoch=None,
    )
    if not validation_only:
        trainer.fit_loop.epoch_loop = _UpdateValidationLoop(trainer, every)
    return trainer


def checkpoint_callbacks(config: DictConfig, module: TrainingModule) -> list[Callback]:
    checkpoint = config.checkpoint
    directory = Path(config.output_dir) / "checkpoints"
    callbacks: list[Callback] = []
    if checkpoint.periodic_every_updates is not None:
        if checkpoint.periodic_every_updates < 1 or checkpoint.keep_periodic < -1:
            raise ValueError("periodic checkpoint interval must be positive and retention >= -1")
        callbacks.append(_UpdateCheckpoint(
            dirpath=directory, filename="step-{step}", auto_insert_metric_name=False,
            monitor="step", mode="max", every_n_train_steps=checkpoint.periodic_every_updates,
            every_n_epochs=0, save_top_k=checkpoint.keep_periodic, save_last=checkpoint.save_last,
            save_on_train_epoch_end=False,
        ))
    if config.validation.every_updates is not None and config.validation.limit_batches != 0:
        callbacks.insert(0, build_validation_checkpoint(
            module, dirpath=directory, filename="best-{step}", auto_insert_metric_name=False,
            save_top_k=checkpoint.save_top_k, save_last=False,
            every_n_epochs=1, every_n_train_steps=0, save_on_train_epoch_end=False,
        ))
    return callbacks


def initialize_model(config: DictConfig, model: NativeModel, tokenizer: PreTrainedTokenizerBase,
                tokens: NativeTokenSchema) -> InitializationReport:
    from training.checkpoint import load_native_checkpoint
    from weight_adapters import import_official_mdlm, initialize_native_family
    startup = config.startup
    if startup.source_kind == "native":
        source_tokenizer = checkpoint_tokenizer(startup)
        check_vocabulary(tokenizer, tokens, source_tokenizer)
        source = load_native_checkpoint(startup.path, weights=startup.weights, token_schema=tokens)
        return initialize_native_family(
            source.model, model, seed=config.seed,
            source_reference=str(source.checkpoint_path),
        )
    if startup.source_kind == "mdlm":
        if startup.weights != "raw":
            raise ValueError("official safetensors initialization provides raw weights")
        source_tokenizer = load_tokenizer(startup.source_tokenizer_path or startup.path)
        return import_official_mdlm(
            startup.path, model, source_tokenizer=source_tokenizer,
            target_tokenizer=tokenizer, target_tokens=tokens,
            config_path=startup.source_config, seed=config.seed,
            source_mask_token_id=startup.source_mask_token_id,
        )
    raise ValueError("startup.source_kind must be native or mdlm")


def check_startup(config: DictConfig, *, modes: Sequence[str]) -> None:
    """Check initialization/resume inputs before creating training state."""
    startup = config.startup
    if startup.mode not in modes:
        raise ValueError("startup.mode must be one of " + ", ".join(modes))
    if startup.mode == "fresh" and startup.path is not None:
        raise ValueError("fresh training must not specify a checkpoint path")
    if startup.mode != "fresh":
        require_path("startup.path", startup.path)
        if not Path(startup.path).exists():
            raise FileNotFoundError("startup.path must identify the requested checkpoint or snapshot")
    if startup.mode == "resume" and (startup.weights != "raw" or startup.source_kind != "native"):
        raise ValueError("resume loads the complete native state, not selected or imported weights")
    output = Path(config.output_dir)
    if startup.mode == "resume" and Path(startup.path).resolve().parent != (output / "checkpoints").resolve():
        raise ValueError("resume must continue the checkpoint's original output_dir")
    if startup.mode != "resume" and ((output / "tokenizer").exists()
                                     or any((output / "checkpoints").glob("*.ckpt"))):
        raise FileExistsError("choose a new output_dir for a new training run")


def fit_prepared(config: DictConfig, module: TrainingModule, tokenizer: PreTrainedTokenizerBase,
         train: PreparedDataset | None, valid: PreparedDataset, *, metadata: Callback,
         initialization: InitializationReport | None = None) -> Trainer:
    """Run a prepared task with shared save, validation and resume callbacks."""
    from training.resume import (
        LayoutResumeCheckpointIO,
        ResumeLayout,
        ResumeMetadata,
        ResumeRuntime,
    )

    final_path = Path(config.output_dir) / "checkpoints" / ("last.ckpt" if config.checkpoint.save_last else "final.ckpt")
    callbacks: list[Callback] = [metadata, ResumeMetadata(),
                 _RunOutput(config, tokenizer, initialization=initialization, final_path=final_path)]
    callbacks.extend(checkpoint_callbacks(config, module))
    trainer = build_workflow_trainer(config, callbacks)
    trainer.strategy.checkpoint_io = LayoutResumeCheckpointIO(
        ResumeLayout(config.runtime.local_batch_size, trainer.num_devices * trainer.num_nodes,
                     trainer.accumulate_grad_batches),
        runtime=ResumeRuntime(trainer.precision, trainer.gradient_clip_val,
                              trainer.gradient_clip_algorithm),
    )
    fit_token_model(module, data_module(config, train, valid), trainer=trainer,
                    ckpt_path=config.startup.path if config.startup.mode == "resume" else None)
    return trainer


def validation_options(config: DictConfig, source: NativeCheckpointLoad) -> tuple[dict[str, Any], bool]:
    """Select saved objective settings and the requested validation weights."""
    selection = config.validation.weights
    if selection not in ("auto", "raw", "ema"):
        raise ValueError("validation.weights must be auto, raw or ema")
    use_ema = selection == "ema" or (selection == "auto" and source.ema_state is not None)
    if use_ema and source.ema_state is None:
        raise ValueError("checkpoint has no EMA weights")
    options = typed_training_options(source.training_contract, source.family)
    if not use_ema:
        options["ema_decay"] = None
    if source.family == "dadlm":
        options.update(graph_backend=config.graph.backend,
                       warmup_before_training=config.graph.cuda.warmup_before_training,
                       monitor_coherence=config.training.monitor_coherence)
    return options, use_ema


def validate_prepared(
    config: DictConfig, source: NativeCheckpointLoad,
    tokenizer: PreTrainedTokenizerBase, tokens: NativeTokenSchema,
    train: PreparedDataset | None, valid: PreparedDataset,
    *, build_module: Callable[..., TrainingModule],
    module_args: tuple[Any, ...] = (),
) -> ValidationResults:
    """Validate a checked task using its saved objective and selected weights."""
    check_vocabulary(tokenizer, tokens, checkpoint_tokenizer(config.startup))
    if source.tokens != tokens:
        raise ValueError("validation tokenizer differs from checkpoint token roles")
    options, use_ema = validation_options(config, source)
    module = build_module(source.model, tokens, *module_args, **options)
    if use_ema:
        cast(ExponentialMovingAverage, module.ema).load_state_dict(
            cast(Mapping[str, Any], source.ema_state))
    trainer = build_workflow_trainer(config, [], validation_only=True)
    return validate_token_model(module, data_module(config, train, valid), trainer=trainer)
