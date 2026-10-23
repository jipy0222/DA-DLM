"""Import published MDLM and full-context BD3LM-pretraining snapshots.

Both releases use the same bidirectional MDLM weight layout. A local snapshot
consists of ``config.json`` and ``model.safetensors``; importing it does not
load publisher Python.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Tuple

import torch

from data.tokens import NativeTokenSchema
from data.types import Metadata
from models.backbone.config import ConditioningMode
from models.mdlm import MDLMModel

from .vocabulary import VocabularyMapping as VocabularyMapping
from .vocabulary import copy_vocab_tensor as _copy_vocab_tensor

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


IMPORTER_VERSION = "kuleshov-mdlm-v3"
VOCAB_SOURCE_KEYS = frozenset(
    {
        "backbone.vocab_embed.embedding",
        "backbone.output_layer.linear.weight",
        "backbone.output_layer.linear.bias",
    }
)
OPTIONAL_SOURCE_METADATA = ("sampling_eps_min", "sampling_eps_max")


@dataclass(frozen=True)
class SourceProvenance:
    repo: Optional[str] = None
    revision: Optional[str] = None
    importer_version: str = IMPORTER_VERSION


BD3LM_PRETRAIN_PROVENANCE = SourceProvenance(
    repo="kuleshov-group/bd3lm-owt-block_size1024-pretrain",
    revision="745abdce30b29e338dc50467422cf70a4e69cfd0",
)


@dataclass(frozen=True)
class MDLMSourceConfig:
    """Architecture fields read from a published snapshot's ``config.json``."""

    hidden_dim: int
    cond_dim: int
    n_blocks: int
    n_heads: int
    model_length: int
    dropout: float
    time_conditioning: bool
    vocab_size: int
    torch_dtype: str
    model_type: str = "mdlm"
    block_size: Optional[int] = None
    var_min: bool = False
    sampling_eps_min: Optional[float] = None
    sampling_eps_max: Optional[float] = None

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "MDLMSourceConfig":
        required = {
            "hidden_dim",
            "cond_dim",
            "n_blocks",
            "n_heads",
            "model_length",
            "dropout",
            "time_conditioning",
            "vocab_size",
            "torch_dtype",
        }
        missing = sorted(required - set(config))
        if missing:
            raise ValueError("source config is missing required fields: %s" % missing)
        model_type = config.get("model_type")
        if model_type not in ("mdlm", "bd3lm"):
            raise ValueError("source config model_type must be 'mdlm' or 'bd3lm'")
        if model_type == "bd3lm":
            # These switches establish MDLM semantics, not merely matching shapes.
            if (config.get("adaln") is not True
                    or config.get("causal") is not False
                    or config.get("cross_attn") is not False
                    or config.get("block_size") != config["model_length"]):
                raise ValueError("only the full-context bidirectional BD3LM pretraining profile is supported")
        return cls(
            hidden_dim=config["hidden_dim"],
            cond_dim=config["cond_dim"],
            n_blocks=config["n_blocks"],
            n_heads=config["n_heads"],
            model_length=config["model_length"],
            dropout=float(config["dropout"]),
            time_conditioning=config["time_conditioning"],
            vocab_size=config["vocab_size"],
            torch_dtype=config["torch_dtype"],
            model_type=model_type,
            block_size=config.get("block_size") if model_type == "bd3lm" else None,
            var_min=config.get("var_min", False) if model_type == "bd3lm" else False,
            sampling_eps_min=config.get("sampling_eps_min") if model_type == "bd3lm" else None,
            sampling_eps_max=config.get("sampling_eps_max") if model_type == "bd3lm" else None,
        )

    @classmethod
    def from_json(cls, path: Path) -> "MDLMSourceConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise TypeError("source config JSON must contain an object")
        return cls.from_mapping(value)

    def validate_target(self, model: MDLMModel) -> None:
        target = model.config
        actual = {
            "hidden_dim": target.backbone.hidden_size,
            "cond_dim": target.backbone.cond_dim,
            "n_blocks": target.backbone.n_layers,
            "n_heads": target.backbone.n_heads,
            "model_length": target.backbone.trained_context_length,
            "dropout": target.backbone.dropout,
        }
        expected = {
            "hidden_dim": self.hidden_dim,
            "cond_dim": self.cond_dim,
            "n_blocks": self.n_blocks,
            "n_heads": self.n_heads,
            "model_length": self.model_length,
            "dropout": self.dropout,
        }
        if actual != expected:
            raise ValueError(
                "source/target architecture mismatch: source=%s target=%s"
                % (expected, actual)
            )
        if self.time_conditioning:
            raise ValueError("MDLM baseline compatibility requires time_conditioning=false")
        if self.torch_dtype != "float32":
            raise ValueError("MDLM pretraining source torch_dtype must be float32")


def build_tokenizer_vocabulary_mapping(
    source_tokenizer: PreTrainedTokenizerBase,
    target_tokenizer: PreTrainedTokenizerBase,
    target_tokens: NativeTokenSchema,
    *,
    source_vocab_size: int,
    source_mask_token_id: Optional[int] = None,
) -> VocabularyMapping:
    """Map official MASK/PAD roles semantically and all other rows by token text."""

    if type(source_vocab_size) is not int or source_vocab_size <= 0:
        raise ValueError("source_vocab_size must be a positive integer")
    if not isinstance(target_tokens, NativeTokenSchema):
        raise TypeError("target_tokens must be a NativeTokenSchema")
    if len(target_tokenizer) != target_tokens.vocab_size:
        raise ValueError("target tokenizer size differs from its token schema")
    for role in ("mask", "pad", "bos", "eos"):
        actual = getattr(target_tokenizer, role + "_token_id", None)
        expected = getattr(target_tokens, role + "_token_id")
        if actual != expected:
            raise ValueError("target tokenizer %s ID differs from its token schema" % role)

    source_vocab = source_tokenizer.get_vocab()
    target_vocab = target_tokenizer.get_vocab()
    inverse: dict[int, str] = {}
    for token, row in source_vocab.items():
        if 0 <= row < source_vocab_size:
            if row in inverse:
                raise ValueError("source tokenizer has duplicate vocabulary row %d" % row)
            inverse[row] = token

    role_rows: dict[int, tuple[int, str]] = {}
    tokenizer_mask = getattr(source_tokenizer, "mask_token_id", None)
    if source_mask_token_id is not None:
        if type(source_mask_token_id) is not int:
            raise TypeError("source_mask_token_id must be an integer")
        if not 0 <= source_mask_token_id < source_vocab_size:
            raise ValueError("source_mask_token_id is outside the source vocabulary")
        if tokenizer_mask is not None and tokenizer_mask != source_mask_token_id:
            raise ValueError(
                "source_mask_token_id differs from the source tokenizer MASK ID"
            )
        source_mask = source_mask_token_id
    else:
        source_mask = tokenizer_mask
    if source_mask is not None and 0 <= source_mask < source_vocab_size:
        role_rows[source_mask] = (target_tokens.mask_token_id, "mask")
    source_pad = getattr(source_tokenizer, "pad_token_id", None)
    aliases = {
        source_mask,
        getattr(source_tokenizer, "bos_token_id", None),
        getattr(source_tokenizer, "eos_token_id", None),
    }
    if (source_pad is not None and 0 <= source_pad < source_vocab_size
            and source_pad not in aliases):
        role_rows[source_pad] = (target_tokens.pad_token_id, "pad")

    rows: dict[int, int] = {}
    semantics: dict[int, str] = {}
    for source_row in range(source_vocab_size):
        if source_row in role_rows:
            target_row, role = role_rows[source_row]
            semantics[source_row] = role
        else:
            token = inverse.get(source_row)
            if token is None or token not in target_vocab:
                raise ValueError("source vocabulary row %d has no target token-text match" % source_row)
            target_row = target_vocab[token]
            if target_row in target_tokens.blocked_token_ids:
                raise ValueError("ordinary source token maps to a target control role")
        rows[source_row] = target_row
    mapping = VocabularyMapping(rows, semantics=semantics)
    mapping.validate(source_vocab_size, target_tokens.vocab_size)
    return mapping


@dataclass(frozen=True)
class MDLMImportReport:
    provenance: Optional[SourceProvenance]
    source_entry_count: int
    consumed_source_keys: Tuple[str, ...]
    target_keys: Tuple[str, ...]
    vocabulary_rows: Tuple[Tuple[int, int, str], ...]
    discarded_source_rows: Tuple[Tuple[int, str], ...]
    initialized_target_rows: Tuple[int, ...]
    source_config: Optional[Metadata]
    target_config: Metadata
    target_dtype: str
    non_backbone_buffers: Tuple[Tuple[str, float], ...]
    source_to_target_keys: Tuple[Tuple[str, str], ...]
    adapter_version: str = IMPORTER_VERSION

    def to_dict(self) -> Metadata:
        return asdict(self)


def load_safetensors_state(path: Path) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(str(Path(path)), device="cpu")


def mdlm_key_map(n_layers: int) -> Dict[str, str]:
    """Map the published DIT state names to native MDLM state names."""

    mapping = {
        "backbone.vocab_embed.embedding": "token_embedding.weight",
        "backbone.sigma_map.mlp.0.weight": "conditioner.mlp.0.weight",
        "backbone.sigma_map.mlp.0.bias": "conditioner.mlp.0.bias",
        "backbone.sigma_map.mlp.2.weight": "conditioner.mlp.2.weight",
        "backbone.sigma_map.mlp.2.bias": "conditioner.mlp.2.bias",
        "backbone.output_layer.norm_final.weight": "core.final_norm.weight",
        "backbone.output_layer.adaLN_modulation.weight": "core.final_ada_ln.weight",
        "backbone.output_layer.adaLN_modulation.bias": "core.final_ada_ln.bias",
        "backbone.output_layer.linear.weight": "vocabulary_head.weight",
        "backbone.output_layer.linear.bias": "vocabulary_head.bias",
    }
    block_suffixes = {
        "norm1.weight": "norm1.weight",
        "attn_qkv.weight": "attention.qkv.weight",
        "attn_out.weight": "attention.out.weight",
        "norm2.weight": "norm2.weight",
        "mlp.0.weight": "mlp.0.weight",
        "mlp.0.bias": "mlp.0.bias",
        "mlp.2.weight": "mlp.2.weight",
        "mlp.2.bias": "mlp.2.bias",
        "adaLN_modulation.weight": "ada_ln_modulation.weight",
        "adaLN_modulation.bias": "ada_ln_modulation.bias",
    }
    for layer in range(n_layers):
        for source_suffix, target_suffix in block_suffixes.items():
            mapping[
                "backbone.blocks.%d.%s" % (layer, source_suffix)
            ] = "core.blocks.%d.%s" % (layer, target_suffix)
    return mapping


def import_mdlm_state(
    model: MDLMModel,
    source_state: Mapping[str, torch.Tensor],
    *,
    vocabulary: VocabularyMapping,
    provenance: Optional[SourceProvenance] = None,
    source_config: Optional[MDLMSourceConfig] = None,
) -> MDLMImportReport:
    """Load one published state dict with an explicit vocabulary-row mapping.

    The state dict contains every mapped backbone parameter and the saved RoPE
    inverse-frequency buffer. Publisher sampling bounds may also be present and
    are copied only to the report.
    """

    if not isinstance(model, MDLMModel):
        raise TypeError("import into native MDLM first; perform family up-training initialization separately")
    if model.config.tie_word_embeddings:
        raise ValueError("the MDLM pretraining importer requires the untied compatibility profile")
    if (model.config.conditioning is not ConditioningMode.ZERO_TIME_COMPAT
            or model.config.timestep_embedding_size != 256):
        raise ValueError("the source requires ZERO_TIME_COMPAT and timestep_embedding_size=256")
    if source_config is not None:
        source_config.validate_target(model)
    mapping = mdlm_key_map(model.config.backbone.n_layers)
    expected = frozenset(mapping) | {"backbone.rotary_emb.inv_freq"}
    actual = frozenset(source_state)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected - frozenset(OPTIONAL_SOURCE_METADATA))
    if missing or unexpected:
        raise ValueError(
            "source manifest mismatch; missing=%s unexpected=%s"
            % (missing, unexpected)
        )
    wrong_dtype = sorted(
        key for key, tensor in source_state.items()
        if key not in OPTIONAL_SOURCE_METADATA and tensor.dtype is not torch.float32
    )
    if wrong_dtype:
        raise TypeError("MDLM pretraining source tensors must be float32: %s" % wrong_dtype)
    target_state = model.state_dict()
    if any(tensor.dtype is not torch.float32 for tensor in target_state.values()):
        raise TypeError("import into FP32 target parameters before enabling mixed-precision execution")
    non_backbone_buffers = [
        (name, source_state[name].item())
        for name in OPTIONAL_SOURCE_METADATA if name in actual
    ]
    expected_targets = frozenset(mapping.values())
    if frozenset(target_state) != expected_targets:
        raise ValueError(
            "native MDLM state manifest mismatch; missing=%s unexpected=%s"
            % (
                sorted(expected_targets - frozenset(target_state)),
                sorted(frozenset(target_state) - expected_targets),
            )
        )

    inverse_frequency = source_state["backbone.rotary_emb.inv_freq"]
    native_inverse_frequency = model.core.rotary.inv_freq.detach().cpu()
    if inverse_frequency.shape != native_inverse_frequency.shape or not torch.equal(
        inverse_frequency.cpu(), native_inverse_frequency
    ):
        raise ValueError("source RoPE inverse frequency/base is incompatible")

    source_embedding = source_state["backbone.vocab_embed.embedding"]
    if source_embedding.ndim != 2:
        raise ValueError("source vocabulary embedding must be a matrix")
    source_vocab_size = (source_config.vocab_size if source_config is not None
                         else source_embedding.shape[0])
    for source_key in VOCAB_SOURCE_KEYS:
        source_tensor = source_state[source_key]
        target_tensor = target_state[mapping[source_key]]
        trailing_shape = () if source_key.endswith(".bias") else (model.config.backbone.hidden_size,)
        if source_tensor.shape != (source_vocab_size,) + trailing_shape:
            raise ValueError("source vocabulary tensor/config mismatch for %s" % source_key)
        if target_tensor.shape != (model.config.vocab_size,) + trailing_shape:
            raise ValueError("target vocabulary tensor/config shape mismatch for %s" % source_key)
    vocabulary.validate(source_vocab_size, model.config.vocab_size)
    vocabulary_rows = tuple(
        (source, target, vocabulary.semantics.get(source, ""))
        for source, target in sorted(vocabulary.source_to_target.items())
    )
    transformed = {}
    for source_key, target_key in mapping.items():
        source_tensor = source_state[source_key]
        target_tensor = target_state[target_key]
        if source_key in VOCAB_SOURCE_KEYS:
            transformed[target_key] = _copy_vocab_tensor(
                source_tensor, target_tensor, vocabulary
            )
        else:
            if source_tensor.shape != target_tensor.shape:
                raise ValueError(
                    "shape mismatch for %s -> %s: %s != %s"
                    % (source_key, target_key, source_tensor.shape, target_tensor.shape)
                )
            transformed[target_key] = source_tensor.to(dtype=target_tensor.dtype)

    model.load_state_dict(transformed, strict=True)
    initialized_rows = tuple(
        sorted(set(range(model.config.vocab_size)) - set(vocabulary.source_to_target.values()))
    )
    return MDLMImportReport(
        provenance=provenance,
        source_entry_count=len(source_state),
        consumed_source_keys=tuple(sorted(source_state)),
        target_keys=tuple(sorted(transformed)),
        vocabulary_rows=vocabulary_rows,
        discarded_source_rows=tuple(sorted(vocabulary.discarded_source.items())),
        initialized_target_rows=initialized_rows,
        source_config=asdict(source_config) if source_config is not None else None,
        target_config=asdict(model.config),
        target_dtype="float32",
        non_backbone_buffers=tuple(non_backbone_buffers),
        source_to_target_keys=tuple(sorted(mapping.items())),
    )


def load_mdlm_checkpoint(
    snapshot_directory: Path,
    model: MDLMModel,
    *,
    vocabulary: VocabularyMapping,
    provenance: Optional[SourceProvenance] = None,
) -> MDLMImportReport:
    """Load ``config.json`` and ``model.safetensors`` from a local HF snapshot."""

    snapshot_directory = Path(snapshot_directory)
    source_config = MDLMSourceConfig.from_json(snapshot_directory / "config.json")
    source_state = load_safetensors_state(snapshot_directory / "model.safetensors")
    return import_mdlm_state(
        model,
        source_state,
        vocabulary=vocabulary,
        provenance=provenance,
        source_config=source_config,
    )
