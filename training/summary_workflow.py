"""Offline summary preparation, fine-tuning and checkpoint loss validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

from lightning import seed_everything
from lightning.pytorch import Trainer
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from data.summary import SummaryConfig
from data.summary_storage import (
    load_summary_store,
    prepare_summary_store,
    summary_store_metadata,
)
from data.tokenizer_io import check_vocabulary as _check_vocabulary
from data.tokenizer_io import load_tokenizer as _tokenizer
from data.tokens import NativeTokenSchema, prepare_tokenizer
from training._types import ValidationResults, WorkflowResult
from training.checkpoint import NativeCheckpointMetadata, load_native_checkpoint
from training.checkpoint import checkpoint_tokenizer as _checkpoint_tokenizer
from training.configuration import model_config as model_config
from training.configuration import require_path as _require_path
from training.configuration import training_options as training_options
from training.runtime import build_workflow_trainer as _trainer  # noqa: F401
from training.runtime import check_startup as _check_startup
from training.runtime import fit_prepared as _fit
from training.runtime import initialize_model as _initialize
from training.runtime import require_single_preparation as _single_preparation
from training.runtime import validate_prepared
from training.setup import build_model, build_summary_module

if TYPE_CHECKING:
    from datasets import Dataset


def _summary_config(data: DictConfig) -> SummaryConfig:
    return SummaryConfig(
        max_source_tokens=data.max_source_tokens,
        max_target_tokens=data.max_target_tokens,
        source_field=data.source_field, reference_field=data.reference_field,
    )


def _metadata(config: DictConfig, tokens: NativeTokenSchema, *,
              task: Mapping[str, Any] | None = None) -> dict[str, Any]:
    data = config.data
    summary = _summary_config(data) if task is None else SummaryConfig(**task["summary"])
    fields = ("profile", "revision", "train_split", "validation_split")
    options = {name: getattr(data, name) if task is None else task[name] for name in fields}
    return summary_store_metadata(
        tokens, summary, tokenizer_id=data.tokenizer_id, **options)


def prepare_summary(config: DictConfig) -> dict[str, int | str]:
    """Persist both prepared splits and their tokenizer before training starts."""
    data = config.data
    _require_path("data.corpus_path", data.corpus_path)
    _require_path("data.tokenizer_path", data.tokenizer_path)
    _single_preparation()
    tokenizer = _tokenizer(data.tokenizer_path)
    tokens = prepare_tokenizer(tokenizer)
    tokenizer_dir = Path(data.prepared_tokenizer_dir)
    if tokenizer_dir.exists():
        _check_vocabulary(tokenizer, tokens, _tokenizer(tokenizer_dir))
    summary = _summary_config(data)
    train, valid = prepare_summary_store(
        data.corpus_path, tokenizer, tokens, summary,
        store_dir=data.store_dir, profile=data.profile, revision=data.revision,
        tokenizer_id=data.tokenizer_id, train_split=data.train_split,
        validation_split=data.validation_split,
        preparation_num_workers=data.preparation_num_workers,
    )
    if not tokenizer_dir.exists():
        tokenizer.save_pretrained(tokenizer_dir)
    result: dict[str, int | str] = {"train_rows": len(train), "validation_rows": len(valid),
              "source_length": summary.max_source_tokens,
              "target_length": summary.max_target_tokens,
              "tokenizer": str(tokenizer_dir)}
    print(json.dumps(result, indent=2))
    return result


def _prepared_summary(config: DictConfig, *, task: Mapping[str, Any] | None = None,
                      validation_only: bool = False) -> tuple[
    PreTrainedTokenizerBase, NativeTokenSchema, Dataset | None, Dataset, dict[str, Any],
]:
    tokenizer = _tokenizer(config.data.prepared_tokenizer_dir)
    tokens = prepare_tokenizer(tokenizer)
    metadata = _metadata(config, tokens, task=task)
    train, valid = load_summary_store(
        config.data.store_dir, expected_metadata=metadata,
        validation_only=validation_only)
    return tokenizer, tokens, train, valid, metadata["task"]


def finetune(config: DictConfig) -> Trainer:
    """Initialize or resume summary training."""
    _check_startup(config, modes=("initialize", "resume"))
    seed_everything(config.seed, workers=True)
    tokenizer, tokens, train, valid, task = _prepared_summary(config)
    if config.startup.mode == "resume":
        _check_vocabulary(tokenizer, tokens, _checkpoint_tokenizer(config.startup))
    summary = SummaryConfig(**task["summary"])
    model = build_model(
        model_config(config, tokens), tokens,
        context_length=summary.sequence_length, target_length=summary.max_target_tokens)
    initialization = (_initialize(config, model, tokenizer, tokens)
                      if config.startup.mode == "initialize" else None)
    module = build_summary_module(
        model, tokens, summary, **training_options(config, config.model.family))
    return _fit(config, module, tokenizer, train, valid,
                metadata=NativeCheckpointMetadata(tokens, task=task),
                initialization=initialization)


def validate_summary(config: DictConfig) -> ValidationResults:
    """Use a checkpoint's saved task, model and objective on prepared validation rows."""
    _require_path("startup.path", config.startup.path)
    seed_everything(config.seed, workers=True)
    source = load_native_checkpoint(config.startup.path, weights="raw")
    if source.task is None or source.task["kind"] != "summary":
        raise ValueError("summary validation requires a summary checkpoint with task metadata")
    tokenizer, tokens, train, valid, task = _prepared_summary(
        config, task=source.task, validation_only=True)
    return validate_prepared(
        config, source, tokenizer, tokens, train, valid,
        build_module=build_summary_module,
        module_args=(SummaryConfig(**task["summary"]),))



def run(config: DictConfig) -> WorkflowResult:
    """Dispatch summary preparation, fine-tuning and validation."""
    actions: dict[str, Callable[[DictConfig], WorkflowResult]] = {
        "prepare_summary": prepare_summary, "finetune": finetune,
        "validate_summary": validate_summary}
    return actions[config.mode](config)
