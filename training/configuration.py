"""Typed model/training recipes and configuration boundary checks."""

from __future__ import annotations

from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf

from data.tokens import NativeTokenSchema
from models import ARConfig, BD3LMConfig, DADLMConfig, MDLMConfig
from models.backbone.config import (
    AttentionBackend,
    ConditioningMode,
    TransformerConfig,
    TransitionNormalization,
)
from objectives.noise import MaskingSchedule
from training._types import (
    NativeConfig,
)
from training.diffusion import DiffusionTrainingConfig
from training.lr_schedules import LRScheduleConfig
from training.token import (
    AdamWConfig,
)


def plain_config(value: Any) -> Any:
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def require_path(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty path")


def model_config(config: DictConfig, tokens: NativeTokenSchema) -> NativeConfig:
    """Translate one family recipe into the existing typed model configuration."""
    values = plain_config(config.model)
    family = values["family"]
    backbone = TransformerConfig(
        hidden_size=values["hidden_size"], n_heads=values["n_heads"], n_layers=values["n_layers"],
        cond_dim=values["cond_dim"] if family != "ar" else None, ada_ln=family != "ar",
        dropout=values["dropout"], mlp_ratio=values["mlp_ratio"],
        rope_base=values["rope_base"],
        trained_context_length=values["trained_context_length"],
    )
    common: dict[str, Any] = dict(backbone=backbone, vocab_size=tokens.vocab_size,
                  pad_token_id=tokens.pad_token_id,
                  tie_word_embeddings=values["tie_word_embeddings"],
                  default_backend=AttentionBackend(values["attention_backend"]),
                  max_generation_length=values["max_generation_length"])
    if family == "ar":
        return ARConfig(**common)
    common.update(conditioning=ConditioningMode(values["conditioning"]),
                  timestep_embedding_size=values["timestep_embedding_size"])
    if family == "mdlm":
        return MDLMConfig(**common)
    common["block_size"] = values["block_size"]
    if family == "bd3lm":
        return BD3LMConfig(**common)
    if family == "dadlm":
        return DADLMConfig(**common, dag_up_ratio=values["dag_up_ratio"], grouped=values["grouped"],
                           transition_normalization=TransitionNormalization(values["transition_normalization"]))
    raise ValueError("model.family must be ar, mdlm, bd3lm or dadlm")


def typed_training_options(values: Mapping[str, Any], family: str) -> dict[str, Any]:
    """Convert saved or configured training settings without modifying them."""
    optimizer = dict(values["optimizer"])
    optimizer["betas"] = tuple(optimizer["betas"])
    options: dict[str, Any] = dict(optimizer=AdamWConfig(**optimizer),
                   schedule=LRScheduleConfig(**values["schedule"]), ema_decay=values["ema_decay"])
    if family != "ar":
        diffusion = dict(values["diffusion"])
        diffusion["noise"] = MaskingSchedule(**diffusion["noise"])
        options["diffusion"] = DiffusionTrainingConfig(**diffusion)
    if family == "dadlm":
        options["assignment"] = values["assignment"]
    return options


def training_options(config: DictConfig, family: str) -> dict[str, Any]:
    """Resolve training settings and attach the selected graph runtime."""
    options = typed_training_options(plain_config(config.training), family)
    if family == "dadlm":
        options.update(monitor_coherence=config.training.monitor_coherence,
                       graph_backend=config.graph.backend,
                       warmup_before_training=config.graph.cuda.warmup_before_training)
    return options
