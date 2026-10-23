"""Offline construction and weight selection for native training checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Union, cast

import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from omegaconf import DictConfig
from torch.serialization import MAP_LOCATION
from transformers import PreTrainedTokenizerBase

from data.tokenizer_io import load_tokenizer
from data.tokens import NativeTokenSchema
from models import (
    ARConfig,
    ARModel,
    BD3LMConfig,
    BD3LMModel,
    DADLMConfig,
    DADLMModel,
    MDLMConfig,
    MDLMModel,
)
from models.backbone.config import (
    AttentionBackend,
    ConditioningMode,
    TransformerConfig,
    TransitionNormalization,
)
from training._types import Checkpoint, NativeConfig, NativeModel, Pathish, WeightsKind
from training.ema import ExponentialMovingAverage

if TYPE_CHECKING:
    from training.token import TokenTrainingModule


NATIVE_PRETRAINING_METADATA = "native_pretraining"
_METADATA_VERSION = 1
_FAMILY_TYPES = {
    "ar": (ARConfig, ARModel, "ar_pretraining"),
    "mdlm": (MDLMConfig, MDLMModel, "mdlm_training"),
    "bd3lm": (BD3LMConfig, BD3LMModel, "bd3lm_training"),
    "dadlm": (DADLMConfig, DADLMModel, "dadlm_training"),
}
_TASK_KINDS = frozenset(("pretraining", "summary"))


def _family_for_model(model: NativeModel) -> str:
    for family, (_, model_type, _) in _FAMILY_TYPES.items():
        if isinstance(model, model_type):
            return family
    raise TypeError("native checkpoints support AR, MDLM, BD3LM and DADLM models")


def _task_mapping(task: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(task, Mapping):
        raise TypeError("task must be a mapping")
    result = deepcopy(dict(task))
    if result.get("kind") not in _TASK_KINDS:
        raise ValueError("task kind must be 'pretraining' or 'summary'")
    return result


class NativeCheckpointMetadata(Callback):
    """Add token-role and task metadata absent from module contracts."""

    def __init__(self, tokens: NativeTokenSchema, *,
                 task: Mapping[str, Any] | None = None) -> None:
        if not isinstance(tokens, NativeTokenSchema):
            raise TypeError("tokens must be a NativeTokenSchema")
        self.tokens = tokens
        self.task = _task_mapping(
            {"kind": "pretraining"} if task is None else task
        )

    def on_save_checkpoint(self, trainer: Trainer, pl_module: LightningModule,
                           checkpoint: Checkpoint) -> None:
        del trainer
        family, training_key = self._validate_module(pl_module)
        checkpoint[NATIVE_PRETRAINING_METADATA] = {
            "version": _METADATA_VERSION,
            "family": family,
            "training_key": training_key,
            "token_schema": self.tokens.to_dict(),
            "task": deepcopy(self.task),
        }

    def on_load_checkpoint(self, trainer: Trainer, pl_module: LightningModule,
                           checkpoint: Checkpoint) -> None:
        del trainer
        family, training_key = self._validate_module(pl_module)
        metadata = checkpoint.get(NATIVE_PRETRAINING_METADATA)
        if not isinstance(metadata, Mapping) or metadata.get("version") != _METADATA_VERSION:
            raise ValueError("checkpoint has no supported native checkpoint metadata")
        if (metadata.get("family") != family
                or metadata.get("training_key") != training_key):
            raise ValueError("checkpoint family metadata differs from the training module")
        if NativeTokenSchema(**dict(metadata["token_schema"])) != self.tokens:
            raise ValueError("checkpoint token schema differs from the training run")
        saved_task = (_task_mapping(metadata["task"])
                      if "task" in metadata else {"kind": "pretraining"})
        if saved_task != self.task:
            raise ValueError("checkpoint task metadata differs from the training run")

    def _validate_module(self, pl_module: LightningModule) -> tuple[str, str]:
        module = cast("TokenTrainingModule[NativeModel]", pl_module)
        family = _family_for_model(module.model)
        training_key = module.checkpoint_key
        if training_key != _FAMILY_TYPES[family][2]:
            raise ValueError("training checkpoint key does not match the model family")
        config = module.model.config
        if (config.vocab_size != self.tokens.vocab_size
                or config.pad_token_id != self.tokens.pad_token_id):
            raise ValueError("token schema does not match the model vocabulary and PAD")
        return family, training_key


@dataclass(frozen=True)
class NativeCheckpointLoad:
    model: NativeModel
    tokens: NativeTokenSchema
    family: str
    weights_kind: WeightsKind
    training_key: str
    training_contract: Mapping[str, Any]
    ema_state: Optional[Mapping[str, Any]]
    checkpoint_path: Path
    task: Optional[Mapping[str, Any]] = None


def _model_config(family: str, value: Mapping[str, Any]) -> NativeConfig:
    if not isinstance(value, Mapping):
        raise TypeError("native model contract must be a mapping")
    config = dict(value)
    backbone = config.get("backbone")
    if not isinstance(backbone, Mapping):
        raise TypeError("native model contract requires a backbone mapping")
    config["backbone"] = TransformerConfig(**dict(backbone))
    config["default_backend"] = AttentionBackend(config["default_backend"])
    if family != "ar":
        config["conditioning"] = ConditioningMode(config["conditioning"])
    if family == "dadlm":
        config["transition_normalization"] = TransitionNormalization(config["transition_normalization"])
    config_type = _FAMILY_TYPES[family][0]
    return config_type(**config)


def _checkpoint_metadata(
    checkpoint: Mapping[str, Any], external_tokens: NativeTokenSchema | None,
) -> tuple[str, str, NativeTokenSchema, dict[str, Any] | None]:
    metadata = checkpoint.get(NATIVE_PRETRAINING_METADATA)
    if metadata is not None:
        if not isinstance(metadata, Mapping) or metadata.get("version") != _METADATA_VERSION:
            raise ValueError("unsupported native checkpoint metadata")
        family = metadata.get("family")
        training_key = metadata.get("training_key")
        if family not in _FAMILY_TYPES or training_key != _FAMILY_TYPES[family][2]:
            raise ValueError("native checkpoint family metadata is inconsistent")
        stored_tokens: NativeTokenSchema | None = NativeTokenSchema(**dict(metadata["token_schema"]))
        if external_tokens is not None and external_tokens != stored_tokens:
            raise ValueError("external token schema differs from checkpoint metadata")
        task = _task_mapping(metadata["task"]) if "task" in metadata else None
        return family, training_key, stored_tokens, task

    candidates = [
        (family, key) for family, (_, _, key) in _FAMILY_TYPES.items()
        if key in checkpoint
    ]
    if len(candidates) != 1:
        raise ValueError("cannot identify one native training contract")
    family, training_key = candidates[0]
    state = checkpoint[training_key]
    tokens_value = state.get("contract", {}).get("tokens") if isinstance(state, Mapping) else None
    stored_tokens = (NativeTokenSchema(**dict(tokens_value))
                     if isinstance(tokens_value, Mapping) else None)
    if stored_tokens is None and external_tokens is None:
        raise ValueError("checkpoint has no token schema; supply token_schema explicitly")
    if stored_tokens is not None and external_tokens is not None and stored_tokens != external_tokens:
        raise ValueError("external token schema differs from checkpoint contract")
    return family, training_key, cast(NativeTokenSchema, stored_tokens or external_tokens), None


def _raw_model_state(checkpoint: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    state = checkpoint.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("native checkpoint has no Lightning state_dict")
    prefix = "model."
    if any(isinstance(name, str) and name.startswith(prefix) for name in state):
        return {
            name[len(prefix):]: tensor for name, tensor in state.items()
            if isinstance(name, str) and name.startswith(prefix)
        }
    return dict(state)


def _select_ema(model: NativeModel, ema_state: Mapping[str, Any] | None) -> None:
    if not isinstance(ema_state, Mapping):
        raise ValueError("checkpoint has no EMA state")
    ema = ExponentialMovingAverage(
        model,
        decay=cast(float, ema_state.get("decay")),
        use_num_updates=ema_state.get("num_updates") is not None,
    )
    ema.load_state_dict(ema_state)
    ema.copy_to(model)


def load_native_checkpoint(
    path: Pathish,
    *,
    weights: WeightsKind = "raw",
    token_schema: Optional[NativeTokenSchema] = None,
    map_location: MAP_LOCATION = "cpu",
) -> NativeCheckpointLoad:
    """Reconstruct a native model and select raw or saved EMA parameters."""

    if weights not in ("raw", "ema"):
        raise ValueError("weights must be 'raw' or 'ema'")
    if token_schema is not None and not isinstance(token_schema, NativeTokenSchema):
        raise TypeError("token_schema must be a NativeTokenSchema")
    path = Path(path).expanduser().resolve(strict=True)
    registered = set(torch.serialization.get_safe_globals())
    safe: list[Union[Callable[..., Any], tuple[Callable[..., Any], str]]] = [
        kind for kind in (AttentionBackend, ConditioningMode, TransitionNormalization)
            if kind not in registered]
    with torch.serialization.safe_globals(safe):
        checkpoint = torch.load(path, map_location=map_location, weights_only=True, mmap=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("native checkpoint must contain a mapping")
    family, training_key, tokens, task = _checkpoint_metadata(
        checkpoint, token_schema
    )
    training_state = checkpoint.get(training_key)
    if not isinstance(training_state, Mapping):
        raise ValueError("native checkpoint has no family training state")
    contract = training_state.get("contract")
    if not isinstance(contract, Mapping) or "model" not in contract:
        raise ValueError("native checkpoint has no model contract")
    config = _model_config(family, contract["model"])
    model = cast(Callable[[NativeConfig], NativeModel], _FAMILY_TYPES[family][1])(config)
    if config.vocab_size != tokens.vocab_size or config.pad_token_id != tokens.pad_token_id:
        raise ValueError("checkpoint token schema differs from its model contract")
    model.load_state_dict(_raw_model_state(checkpoint), strict=True)
    ema_state = training_state.get("ema")
    if weights == "ema":
        _select_ema(model, ema_state)
    return NativeCheckpointLoad(
        model=model,
        tokens=tokens,
        family=family,
        weights_kind=weights,
        training_key=training_key,
        training_contract=contract,
        ema_state=ema_state,
        checkpoint_path=path,
        task=task,
    )


def checkpoint_tokenizer(startup: DictConfig) -> PreTrainedTokenizerBase:
    """Load the explicit source tokenizer or the checkpoint companion directory."""
    path = (Path(startup.source_tokenizer_path) if startup.source_tokenizer_path
            else Path(startup.path).parent.parent / "tokenizer")
    if not path.is_dir():
        raise FileNotFoundError("checkpoint tokenizer is missing; supply startup.source_tokenizer_path")
    return load_tokenizer(path)


__all__ = [
    "NATIVE_PRETRAINING_METADATA",
    "NativeCheckpointLoad",
    "NativeCheckpointMetadata",
    "load_native_checkpoint",
    "checkpoint_tokenizer",
]
