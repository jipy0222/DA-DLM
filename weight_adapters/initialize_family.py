"""Initialize model-family weights with a validated, atomic load.

Prepare and validate off-model tensors before the single strict load. A rejected
mapping/configuration therefore does not partially reset the target model.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

import torch
from torch import nn

from data.tokens import NativeTokenSchema
from data.types import Metadata, Pathish
from models import ARModel, BD3LMModel, DADLMModel, MDLMConfig, MDLMModel
from models.backbone.config import AttentionBackend
from models.types import NativeModel, TensorState

from .import_mdlm import (
    MDLMImportReport,
    MDLMSourceConfig,
    SourceProvenance,
    build_tokenizer_vocabulary_mapping,
    import_mdlm_state,
    load_safetensors_state,
)
from .vocabulary import VOCABULARY_KEYS as VOCABULARY_KEYS
from .vocabulary import VocabularyMapping
from .vocabulary import copy_vocab_tensor as _copy_vocab_tensor

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


DAD_ONLY_KEYS = (
    "slot_embedding.weight",
    "dag_link.position_embedding.weight",
    "dag_link.query.weight",
    "dag_link.query.bias",
    "dag_link.key.weight",
    "dag_link.key.bias",
    "dag_link.gate.weight",
    "dag_link.gate.bias",
)


@dataclass(frozen=True)
class FamilyInitializationReport:
    source_family: str
    target_family: str
    copied_keys: Tuple[str, ...]
    initialized_keys: Tuple[str, ...]
    vocabulary_rows: Tuple[Tuple[int, int, str], ...]
    initialized_vocabulary_rows: Tuple[int, ...]
    seed: int
    source_config: Metadata
    target_config: Metadata
    parameter_dtype: str
    source_reference: Optional[str]
    discarded_source_rows: Tuple[Tuple[int, str], ...]

    def to_dict(self) -> Metadata:
        return asdict(self)


@dataclass(frozen=True)
class OfficialMDLMInitialization:
    import_report: MDLMImportReport
    family_report: Optional[FamilyInitializationReport]


def _validate_shared_config(source: NativeModel, target: NativeModel, vocabulary: VocabularyMapping) -> None:
    # Dropout, execution length, and attention backend do not change these
    # shared tensors. Head, RoPE, and conditioning semantics do.
    left, right = source.config.shared(), target.config.shared()
    fields = ("hidden_size", "n_heads", "n_layers", "cond_dim", "mlp_ratio", "rope_base", "ada_ln")
    different = [name for name in fields
                 if getattr(left.backbone, name) != getattr(right.backbone, name)]
    different += [name for name in ("conditioning", "timestep_embedding_size")
                  if getattr(left, name) != getattr(right, name)]
    if different:
        raise ValueError("shared architecture/semantics mismatch: %s" % different)
    if left.tie_word_embeddings or right.tie_word_embeddings:
        raise ValueError("cross-family initialization requires untied output embeddings")
    if vocabulary.source_to_target.get(left.pad_token_id) != right.pad_token_id:
        raise ValueError("the source PAD must map to the target PAD")
    if isinstance(source, DADLMModel) and isinstance(target, DADLMModel):
        for name in ("block_size", "dag_up_ratio", "grouped", "transition_normalization"):
            if getattr(source.config, name) != getattr(target.config, name):
                raise ValueError("DADLM-to-DADLM initialization changes %s" % name)


def _new_vocabulary_state(target_state: TensorState, seed: int) -> dict[str, torch.Tensor]:
    embedding = torch.empty_like(target_state["token_embedding.weight"])
    generator = torch.Generator(device=embedding.device).manual_seed(seed)
    nn.init.kaiming_uniform_(embedding, a=math.sqrt(5), generator=generator)
    return {
        "token_embedding.weight": embedding,
        "vocabulary_head.weight": torch.zeros_like(target_state["vocabulary_head.weight"]),
        "vocabulary_head.bias": torch.zeros_like(target_state["vocabulary_head.bias"]),
    }


def _new_dad_state(target_state: TensorState, seed: int) -> dict[str, torch.Tensor]:
    # Match nn.Embedding / nn.Linear defaults with device-local generators so
    # initialization does not advance the caller's global RNG streams.
    generators = {}
    result = {}
    for name in DAD_ONLY_KEYS:
        tensor = torch.empty_like(target_state[name])
        if tensor.device not in generators:
            generators[tensor.device] = torch.Generator(device=tensor.device).manual_seed(seed)
        generator = generators[tensor.device]
        if name in DAD_ONLY_KEYS[:2]:
            nn.init.normal_(tensor, generator=generator)
        elif name.endswith("weight"):
            nn.init.kaiming_uniform_(tensor, a=math.sqrt(5), generator=generator)
        else:
            fan_in = target_state[name[:-4] + "weight"].shape[1]
            nn.init.uniform_(tensor, -1 / math.sqrt(fan_in), 1 / math.sqrt(fan_in), generator=generator)
        result[name] = tensor
    return result


def initialize_family_from_shared(
    source: NativeModel,
    target: NativeModel,
    *,
    seed: int,
    vocabulary: Optional[VocabularyMapping] = None,
    source_reference: Optional[str] = None,
) -> FamilyInitializationReport:
    """Copy shared native weights and initialize any target-only DAD modules.

    ``source_reference`` is an optional path or run ID recorded for training
    lineage. Dropout/backend/execution-length changes are permitted and recorded
    in the report, never inferred from weights.
    """
    if vocabulary is None:
        if source.config.vocab_size != target.config.vocab_size:
            raise ValueError("vocabulary expansion requires an explicit semantic mapping")
        vocabulary = VocabularyMapping.identity(source.config.vocab_size)
    vocabulary.validate(source.config.vocab_size, target.config.vocab_size)
    _validate_shared_config(source, target, vocabulary)
    source_state, target_state = source.state_dict(), target.state_dict()
    missing_source = sorted(set(source_state) - set(target_state))
    if missing_source:
        raise ValueError("source parameters were not consumed: %s" % missing_source)
    initialized = sorted(set(target_state) - set(source_state))
    if initialized and (not isinstance(source, BD3LMModel)
                        or not isinstance(target, DADLMModel)):
        raise ValueError("new modules are only supported for BD3LM -> DADLM")
    if initialized and set(initialized) != set(DAD_ONLY_KEYS):
        raise ValueError("unexpected new DADLM parameter manifest")
    for name, tensor in source_state.items():
        other = target_state[name]
        if tensor.dtype != other.dtype:
            raise TypeError("shared parameter dtype mismatch: %s; initialize before casting" % name)
        if name in VOCABULARY_KEYS:
            trailing_shape = () if name.endswith(".bias") else (source.config.backbone.hidden_size,)
            source_shape = (source.config.vocab_size,) + trailing_shape
            target_shape = (target.config.vocab_size,) + trailing_shape
            if tensor.shape != source_shape or other.shape != target_shape:
                raise ValueError("shared vocabulary parameter shape mismatch for %s" % name)
        elif tensor.shape != other.shape:
            raise ValueError("shared parameter shape mismatch for %s" % name)
    prepared = {name: tensor.detach().to(device=target_state[name].device)
                for name, tensor in source_state.items() if name not in VOCABULARY_KEYS}
    new_vocabulary = (_new_vocabulary_state(target_state, seed)
                      if len(vocabulary.source_to_target) < target.config.vocab_size else target_state)
    for name in VOCABULARY_KEYS:
        prepared[name] = _copy_vocab_tensor(source_state[name], new_vocabulary[name], vocabulary)
    if initialized:
        prepared.update(_new_dad_state(target_state, seed))
    # The only mutation of target parameters in this function.
    target.load_state_dict(prepared, strict=True)
    return FamilyInitializationReport(
        source_family=type(source).__name__, target_family=type(target).__name__,
        copied_keys=tuple(sorted(source_state)), initialized_keys=tuple(initialized),
        vocabulary_rows=tuple((s, t, vocabulary.semantics.get(s, ""))
                              for s, t in sorted(vocabulary.source_to_target.items())),
        initialized_vocabulary_rows=tuple(sorted(set(range(target.config.vocab_size))
                                                - set(vocabulary.source_to_target.values()))),
        seed=seed, source_config=asdict(source.config), target_config=asdict(target.config),
        parameter_dtype=str(target.token_embedding.weight.dtype),
        source_reference=source_reference,
        discarded_source_rows=tuple(sorted(vocabulary.discarded_source.items())),
    )


def initialize_dadlm_from_bd3lm(
    source: BD3LMModel,
    target: DADLMModel,
    *,
    seed: int,
    vocabulary: Optional[VocabularyMapping] = None,
    source_reference: Optional[str] = None,
) -> FamilyInitializationReport:
    if not isinstance(source, BD3LMModel) or not isinstance(target, DADLMModel):
        raise TypeError("expected native BD3LM source and DADLM target")
    return initialize_family_from_shared(
        source, target, seed=seed, vocabulary=vocabulary,
        source_reference=source_reference,
    )


def initialize_native_family(
    source_model: NativeModel,
    target_model: NativeModel,
    *,
    seed: int,
    vocabulary: Optional[VocabularyMapping] = None,
    source_reference: Optional[str] = None,
) -> FamilyInitializationReport:
    """Initialize a same-family model, MDLM -> BD3LM, or BD3LM -> DADLM."""
    routes = (
        (ARModel, ARModel), (MDLMModel, MDLMModel),
        (BD3LMModel, BD3LMModel), (DADLMModel, DADLMModel),
        (MDLMModel, BD3LMModel), (BD3LMModel, DADLMModel),
    )
    if not any(isinstance(source_model, source) and isinstance(target_model, target)
               for source, target in routes):
        raise ValueError("supported native initialization is same-family, MDLM -> BD3LM or BD3LM -> DADLM")
    return initialize_family_from_shared(
        source_model, target_model, seed=seed, vocabulary=vocabulary,
        source_reference=source_reference,
    )


def import_official_mdlm(
    snapshot_directory: Pathish,
    target_model: NativeModel,
    *,
    source_tokenizer: PreTrainedTokenizerBase,
    target_tokenizer: PreTrainedTokenizerBase,
    target_tokens: NativeTokenSchema,
    provenance: Optional[SourceProvenance] = None,
    config_path: Pathish | None=None,
    source_mask_token_id: Optional[int] = None,
    seed: int = 0,
) -> OfficialMDLMInitialization:
    """Import local official weights into MDLM or initialize BD3LM through MDLM."""

    if not isinstance(target_model, (MDLMModel, BD3LMModel)):
        raise TypeError("official MDLM weights support native MDLM or BD3LM targets")
    if (target_model.config.vocab_size != target_tokens.vocab_size
            or target_model.config.pad_token_id != target_tokens.pad_token_id):
        raise ValueError("target model and token schema disagree")
    snapshot_directory = Path(snapshot_directory).expanduser().resolve(strict=True)
    config_path = (snapshot_directory / "config.json" if config_path is None else
                   Path(config_path).expanduser().resolve(strict=True))
    source_config = MDLMSourceConfig.from_json(config_path)
    vocabulary = build_tokenizer_vocabulary_mapping(
        source_tokenizer,
        target_tokenizer,
        target_tokens,
        source_vocab_size=source_config.vocab_size,
        source_mask_token_id=source_mask_token_id,
    )
    if isinstance(target_model, MDLMModel):
        imported = target_model
    else:
        config = target_model.config
        imported = MDLMModel(MDLMConfig(
            backbone=config.backbone,
            vocab_size=config.vocab_size,
            pad_token_id=config.pad_token_id,
            conditioning=config.conditioning,
            tie_word_embeddings=False,
            default_backend=AttentionBackend.SDPA,
            timestep_embedding_size=config.timestep_embedding_size,
        ))
        imported.load_state_dict(target_model.state_dict(), strict=True)
    source_state = load_safetensors_state(snapshot_directory / "model.safetensors")
    import_report = import_mdlm_state(
        imported,
        source_state,
        vocabulary=vocabulary,
        provenance=provenance,
        source_config=source_config,
    )
    family_report = None
    if isinstance(target_model, BD3LMModel):
        family_report = initialize_family_from_shared(
            imported,
            target_model,
            seed=seed,
            vocabulary=VocabularyMapping.identity(target_tokens.vocab_size),
            source_reference=str(snapshot_directory),
        )
    return OfficialMDLMInitialization(
        import_report=import_report,
        family_report=family_report,
    )
