"""Thin Lightning Trainer and validation-checkpoint assembly.

The shared token fit and validation entry points own the lifecycle. This module
creates Lightning objects for full-epoch, rank-sharded training.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from lightning.pytorch import Callback, Trainer
from lightning.pytorch.accelerators import Accelerator
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import Logger
from lightning.pytorch.strategies import Strategy
from lightning.pytorch.trainer.connectors.accelerator_connector import _PRECISION_INPUT

from training._types import Pathish, TrainingModule

_VALIDATION_MONITORS = {
    "AR": "val/nll",
    "MDLM": "val/nll",
    "BD3LM": "val/nll",
    "DADLM": "val/loss",
}


def validation_monitor(module_or_name: TrainingModule | str) -> str:
    """Return the validation metric published by a supported training module."""
    name = (module_or_name if isinstance(module_or_name, str)
            else module_or_name.training_name)
    try:
        return _VALIDATION_MONITORS[name.upper()]
    except (AttributeError, KeyError) as exc:
        raise ValueError("unsupported training family: %r" % name) from exc


def build_validation_checkpoint(module_or_name: TrainingModule | str, *,
                                dirpath: Pathish | None, filename: str,
                                **checkpoint_options: Any) -> ModelCheckpoint:
    """Build a validation-metric checkpoint from caller-supplied save policy.

    Leave ``save_weights_only=False`` when the shared fit lifecycle must resume
    optimizer and data-progress state.
    """
    return ModelCheckpoint(
        dirpath=dirpath, filename=filename,
        monitor=validation_monitor(module_or_name), mode="min",
        **checkpoint_options,
    )


def build_trainer(
    *,
    accelerator: str | Accelerator,
    devices: list[int] | str | int,
    strategy: str | Strategy,
    precision: _PRECISION_INPUT,
    max_steps: int,
    accumulate_grad_batches: int,
    logger: Logger | Iterable[Logger] | bool,
    callbacks: Iterable[Callback],
    enable_checkpointing: bool,
    default_root_dir: Pathish | None = None,
    **trainer_options: Any,
) -> Trainer:
    """Construct the Trainer consumed by the shared token lifecycle.

    Training always keeps the complete dataset epoch and the data module's own
    rank sampler. Other Lightning controls remain explicit caller options.
    """
    locked = {"limit_train_batches", "use_distributed_sampler"} & trainer_options.keys()
    if locked:
        raise TypeError("runner owns Trainer option(s): %s" % ", ".join(sorted(locked)))

    callbacks = list(callbacks)
    return Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate_grad_batches,
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=enable_checkpointing,
        default_root_dir=default_root_dir,
        limit_train_batches=1.0,
        use_distributed_sampler=False,
        **trainer_options,
    )


__all__ = ["build_trainer", "build_validation_checkpoint", "validation_monitor"]
