"""Construct family models and connect prepared tasks to training modules."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, cast, overload

from data.summary import SummaryConfig
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
from training._types import NativeConfig, NativeModel, TrainingModule
from training.ar import ARTrainingModule
from training.bd3lm import BD3LMTrainingModule
from training.dadlm import DADLMTrainingModule
from training.diffusion import DiffusionTrainingConfig
from training.mdlm import MDLMTrainingModule


@overload
def build_model(config: ARConfig, tokens: NativeTokenSchema, *, context_length: int,
                target_length: int | None = None,
                total_content_capacity: int | None = None) -> ARModel: ...


@overload
def build_model(config: MDLMConfig, tokens: NativeTokenSchema, *, context_length: int,
                target_length: int | None = None,
                total_content_capacity: int | None = None) -> MDLMModel: ...


@overload
def build_model(config: BD3LMConfig, tokens: NativeTokenSchema, *, context_length: int,
                target_length: int | None = None,
                total_content_capacity: int | None = None) -> BD3LMModel: ...


@overload
def build_model(config: DADLMConfig, tokens: NativeTokenSchema, *, context_length: int,
                target_length: int | None = None,
                total_content_capacity: int | None = None) -> DADLMModel: ...


def build_model(config: NativeConfig, tokens: NativeTokenSchema, *, context_length: int,
                target_length: int | None = None,
                total_content_capacity: int | None = None) -> NativeModel:
    """Construct a model with schema IDs and task-derived execution capacities.

    ``context_length`` is the clean source+target slot width. ``target_length``
    is its target suffix, defaulting to the whole sequence for pretraining.
    ``total_content_capacity`` bounds prompt plus generated target tokens.
    An explicit argument takes precedence over the model recipe; otherwise the
    recipe's ``max_generation_length`` is used, falling back to
    ``trained_context_length``.

    DADLM rounds only the target's training capacity to a complete block.
    Data and EOS positions stay unchanged. Architecture, dropout, attention
    backend, tied embeddings and trained_context_length come from config.
    Import or initialize weights on the returned model before building a
    training module, whose EMA starts from those weights.
    """
    supported = (ARConfig, MDLMConfig, BD3LMConfig, DADLMConfig)
    if not isinstance(config, supported):
        raise TypeError("config must be ARConfig, MDLMConfig, BD3LMConfig or DADLMConfig")
    if target_length is None:
        target_length = context_length
    if total_content_capacity is None:
        total_content_capacity = config.max_generation_length
    if total_content_capacity is None:
        total_content_capacity = config.backbone.trained_context_length
    if (any(type(value) is not int for value in
            (context_length, target_length, total_content_capacity))
            or not 0 < target_length <= context_length or total_content_capacity <= 0):
        raise ValueError("lengths require integers with 0 < target <= context and total capacity > 0")
    config = replace(config, vocab_size=tokens.vocab_size, pad_token_id=tokens.pad_token_id)
    model_type: Callable[..., NativeModel]
    if isinstance(config, (BD3LMConfig, DADLMConfig)):
        if isinstance(config, DADLMConfig):
            padding = (-target_length) % config.block_size
            context_length += padding
            target_length += padding
        config = config.with_lengths(context_length=context_length,
                                     current_length=target_length,
                                     generation_length=total_content_capacity)
        model_type = DADLMModel if isinstance(config, DADLMConfig) else BD3LMModel
    else:
        capacity = max(context_length, total_content_capacity)
        config = replace(
            config,
            backbone=replace(
                config.backbone,
                max_attention_length=capacity,
                max_position_id=capacity - 1,
            ),
            max_generation_length=total_content_capacity,
        )
        model_type = ARModel if isinstance(config, ARConfig) else MDLMModel
    return cast(Callable[[NativeConfig], NativeModel], model_type)(config)


def _check_schema(model: NativeModel, tokens: NativeTokenSchema) -> None:
    if (model.config.vocab_size != tokens.vocab_size
            or model.config.pad_token_id != tokens.pad_token_id):
        raise ValueError("model vocabulary and PAD must match the token schema")


def build_pretraining_module(model: NativeModel, tokens: NativeTokenSchema, *,
                             diffusion: DiffusionTrainingConfig | None = None,
                             **training_options: Any) -> TrainingModule:
    """Connect ordinary BOS and output roles to an initialized model.

    ``training_options`` are forwarded to the family's existing constructor:
    optimizer, schedule, ema_decay, and DADLM assignment/graph options.
    Diffusion settings pass through, including an explicit ignore_bos=False.
    """
    _check_schema(model, tokens)
    if isinstance(model, ARModel):
        if diffusion is not None:
            raise TypeError("AR does not use diffusion settings")
        return ARTrainingModule(model, blocked_token_ids=tokens.blocked_token_ids,
                                **training_options)
    diffusion = diffusion if diffusion is not None else DiffusionTrainingConfig()
    if isinstance(model, DADLMModel):
        return DADLMTrainingModule(model, tokens=tokens, source_length=0, diffusion=diffusion,
                                   **training_options)
    arguments: dict[str, Any] = dict(mask_token_id=tokens.mask_token_id,
                     bos_token_id=tokens.bos_token_id,
                     blocked_token_ids=tokens.blocked_token_ids,
                     diffusion=diffusion, **training_options)
    if isinstance(model, MDLMModel):
        return MDLMTrainingModule(model, **arguments)
    if isinstance(model, BD3LMModel):
        return BD3LMTrainingModule(model, source_length=0, **arguments)
    raise TypeError("unsupported model family")


def build_summary_module(model: NativeModel, tokens: NativeTokenSchema, summary: SummaryConfig, *,
                         diffusion: DiffusionTrainingConfig | None = None,
                         **training_options: Any) -> TrainingModule:
    """Connect prepared summary rows to an initialized model.

    Summary EOS is allowed; source, summary BOS and PAD are protected by the
    prepared loss_mask. MDLM consumes the full sequence starting with source
    text, so its position-zero BOS check is disabled. Block models instead
    check summary BOS at the start of the target suffix. All other diffusion
    settings and family-specific training_options pass through unchanged.
    """
    _check_schema(model, tokens)
    blocked = tuple(index for index in tokens.blocked_token_ids
                    if index != tokens.summary_eos)
    if isinstance(model, ARModel):
        if diffusion is not None:
            raise TypeError("AR does not use diffusion settings")
        return ARTrainingModule(model, blocked_token_ids=blocked, **training_options)
    diffusion = diffusion if diffusion is not None else DiffusionTrainingConfig()
    if isinstance(model, DADLMModel):
        return DADLMTrainingModule(model, tokens=tokens,
                                   source_length=summary.max_source_tokens,
                                   diffusion=diffusion, **training_options)
    arguments: dict[str, Any] = dict(mask_token_id=tokens.mask_token_id, blocked_token_ids=blocked,
                     **training_options)
    if isinstance(model, MDLMModel):
        return MDLMTrainingModule(model, bos_token_id=None,
                                  diffusion=replace(diffusion, ignore_bos=False),
                                  **arguments)
    if isinstance(model, BD3LMModel):
        return BD3LMTrainingModule(model, source_length=summary.max_source_tokens,
                                   bos_token_id=tokens.summary_bos, diffusion=diffusion,
                                   **arguments)
    raise TypeError("unsupported model family")
