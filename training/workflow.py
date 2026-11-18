"""Pretraining data preparation and task assembly."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from lightning import seed_everything
from lightning.pytorch import Trainer
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from data.prepare import prepare_pretraining_datasets
from data.pretrain import PackingConfig
from data.pretrain_storage import TokenDataset
from data.tokenizer_io import check_vocabulary as _check_vocabulary
from data.tokenizer_io import load_tokenizer as _tokenizer
from data.tokens import NativeTokenSchema, prepare_tokenizer
from training._types import (
    ValidationResults,
    WorkflowResult,
)
from training.callbacks import InitializationReport as InitializationReport
from training.callbacks import _initialization_summary as _initialization_summary
from training.callbacks import _RunOutput as _RunOutput
from training.callbacks import _UpdateCheckpoint as _UpdateCheckpoint
from training.callbacks import _UpdateValidationLoop as _UpdateValidationLoop
from training.checkpoint import checkpoint_tokenizer as _checkpoint_tokenizer
from training.configuration import model_config as model_config
from training.configuration import plain_config as _plain  # noqa: F401
from training.configuration import require_path as _require_path
from training.configuration import training_options as training_options
from training.configuration import typed_training_options as _typed_training_options  # noqa: F401
from training.runtime import PreparedDataset as PreparedDataset
from training.runtime import build_logger as _logger  # noqa: F401

# Compatibility exports for callers of the original shared workflow helpers.
from training.runtime import build_workflow_trainer as _trainer  # noqa: F401
from training.runtime import check_startup as _check_startup
from training.runtime import checkpoint_callbacks as _checkpoint_callbacks  # noqa: F401
from training.runtime import data_module as _data_module  # noqa: F401
from training.runtime import fit_prepared as _fit
from training.runtime import initialize_model as _initialize
from training.runtime import require_single_preparation as _single_preparation
from training.runtime import validate_prepared
from training.runtime import validation_options as _validation_options  # noqa: F401
from training.setup import build_model, build_pretraining_module


def _packing(config: DictConfig) -> PackingConfig:
    data = config.data
    return PackingConfig(
        sequence_length=data.sequence_length, pack_batch_size=data.packing_batch_size,
        num_workers=data.packing_num_workers,
        preprocessing="lm1b" if data.profile == "lm1b" else "none",
        document_eos=data.document_eos, window_bos_eos=data.window_bos_eos,
    )


def prepare(config: DictConfig) -> dict[str, int | str]:
    """Build or reuse both token stores before distributed training starts."""
    _require_path("data.corpus_path", config.data.corpus_path)
    _require_path("data.tokenizer_path", config.data.tokenizer_path)
    _single_preparation()
    tokenizer = _tokenizer(config.data.tokenizer_path)
    tokens = prepare_tokenizer(tokenizer)
    tokenizer_dir = Path(config.data.prepared_tokenizer_dir)
    if tokenizer_dir.exists():
        _check_vocabulary(tokenizer, tokens, _tokenizer(tokenizer_dir))
    packing = _packing(config)
    train, valid = prepare_pretraining_datasets(
        config.data.corpus_path, tokenizer, tokens, profile=config.data.profile,
        tokenizer_id=config.data.tokenizer_id,
        train_store_dir=config.data.train_store_dir,
        validation_store_dir=config.data.validation_store_dir,
        train_packing=packing, validation_packing=packing, column=config.data.column,
    )
    if not tokenizer_dir.exists():
        tokenizer.save_pretrained(tokenizer_dir)
    result: dict[str, int | str] = {"train_rows": len(train), "validation_rows": len(valid),
              "sequence_length": packing.sequence_length,
              "tokenizer": str(config.data.prepared_tokenizer_dir)}
    print(json.dumps(result, indent=2))
    return result


def _prepared(config: DictConfig, *, validation_only: bool = False) -> tuple[
    PreTrainedTokenizerBase, NativeTokenSchema, TokenDataset | None, TokenDataset,
]:
    tokenizer = _tokenizer(config.data.prepared_tokenizer_dir)
    tokens = prepare_tokenizer(tokenizer)
    packing = _packing(config).to_metadata()
    paths = [("validation", config.data.validation_store_dir)]
    if not validation_only:
        paths.insert(0, ("train", config.data.train_store_dir))
    datasets: dict[str, TokenDataset] = {}
    for split, path in paths:
        dataset = TokenDataset(path)
        expected = {"special_ids": tokens.to_dict(), "packing": packing,
                    "tokenizer": config.data.tokenizer_id,
                    "split": config.data.profile + ":" + split}
        for key, value in expected.items():
            if dataset.metadata[key] != value:
                raise ValueError("prepared %s data differs in %s" % (split, key))
        datasets[split] = dataset
    return tokenizer, tokens, datasets.get("train"), datasets["validation"]


def pretrain(config: DictConfig) -> Trainer:
    """Assemble one explicit start mode and run the shared training lifecycle."""
    from training.checkpoint import NativeCheckpointMetadata

    _check_startup(config, modes=("fresh", "initialize", "resume"))
    startup = config.startup
    seed_everything(config.seed, workers=True)
    tokenizer, tokens, train, valid = _prepared(config)
    if startup.mode == "resume":
        _check_vocabulary(tokenizer, tokens, _checkpoint_tokenizer(startup))
    model = build_model(model_config(config, tokens), tokens, context_length=config.data.sequence_length)
    initialization = _initialize(config, model, tokenizer, tokens) if startup.mode == "initialize" else None
    module = build_pretraining_module(model, tokens, **training_options(config, config.model.family))
    return _fit(config, module, tokenizer, train, valid,
                metadata=NativeCheckpointMetadata(tokens), initialization=initialization)


def validate(config: DictConfig) -> ValidationResults:
    """Rebuild a native model and objective, without restoring an optimizer."""
    from training.checkpoint import load_native_checkpoint

    _require_path("startup.path", config.startup.path)
    seed_everything(config.seed, workers=True)
    source = load_native_checkpoint(config.startup.path, weights="raw")
    if ((source.task is not None and source.task["kind"] != "pretraining")
            or source.training_contract.get("source_length", 0) != 0):
        raise ValueError("pretraining validation requires a pretraining checkpoint")
    tokenizer, tokens, train, valid = _prepared(config, validation_only=True)
    return validate_prepared(
        config, source, tokenizer, tokens, train, valid,
        build_module=build_pretraining_module)



def run(config: DictConfig) -> WorkflowResult:
    """Dispatch only task assembly; model algorithms stay in their own modules."""
    actions: dict[str, Callable[[DictConfig], WorkflowResult]] = {
        "prepare": prepare, "pretrain": pretrain, "validate": validate}
    return actions[config.mode](config)
