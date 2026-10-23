"""Special-token roles for the model vocabulary.

Register roles on a loaded tokenizer before constructing a model or data cache.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Callable, cast

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


_TASK_TOKEN_NAMES = {
    role: "<|%s|>" % role
    for role in ("block_bos", "block_eos", "summary_bos", "summary_eos")
}


@dataclass(frozen=True)
class NativeTokenSchema:
    vocab_size: int
    pad_token_id: int
    mask_token_id: int
    bos_token_id: int
    eos_token_id: int
    block_bos: int
    block_eos: int
    summary_bos: int
    summary_eos: int

    def __post_init__(self) -> None:
        values = self.to_dict()
        if any(type(value) is not int for value in values.values()):
            raise TypeError("token schema values must be integers")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        ids = [value for key, value in values.items() if key != "vocab_size"]
        if any(value < 0 or value >= self.vocab_size for value in ids):
            raise ValueError("special token IDs must be inside the vocabulary")
        # GPT-2's ordinary BOS/EOS alias is intentional; all other roles differ.
        exclusive = self.blocked_token_ids
        if len(set(exclusive)) != len(exclusive):
            raise ValueError("PAD, MASK and task/graph roles must have distinct IDs")
        if set(exclusive) & {self.bos_token_id, self.eos_token_id}:
            raise ValueError("task/graph/PAD/MASK IDs cannot alias ordinary BOS/EOS")

    @property
    def blocked_token_ids(self) -> tuple[int, ...]:
        """PAD, MASK and task markers excluded from ordinary text targets."""
        return (self.mask_token_id, self.pad_token_id, self.block_bos,
                self.block_eos, self.summary_bos, self.summary_eos)

    def to_dict(self) -> dict[str, int]:
        return cast('dict[str, int]', asdict(self))


def _validate_tokenizer_schema(tokenizer: PreTrainedTokenizerBase, schema: NativeTokenSchema) -> None:
    vocab_size = len(tokenizer)
    standard_ids = (
        tokenizer.bos_token_id, tokenizer.eos_token_id,
        tokenizer.pad_token_id, tokenizer.mask_token_id,
    )
    expected_standard_ids = (
        schema.bos_token_id, schema.eos_token_id,
        schema.pad_token_id, schema.mask_token_id,
    )
    converter = getattr(tokenizer, "convert_tokens_to_ids", None)
    task_ids_match = converter is None or all(
        converter(name) == getattr(schema, role)
        for role, name in _TASK_TOKEN_NAMES.items()
    )
    if (vocab_size != schema.vocab_size
            or standard_ids != expected_standard_ids
            or not task_ids_match):
        raise ValueError(
            "tokenizer vocabulary size or special IDs changed; prepare its schema again"
        )


def prepare_tokenizer(tokenizer: PreTrainedTokenizerBase) -> NativeTokenSchema:
    """Prepare a standard GPT-2/BERT tokenizer in place, before model creation.

    Existing IDs stay unchanged; missing roles receive fixed token names.
    Conflicting roles raise. BERT uses CLS/SEP for BOS/EOS; GPT-2 may share a
    BOS/EOS ID. Repeated preparation is idempotent.
    """
    for role, fallback in (("bos", "cls"), ("eos", "sep")):
        if getattr(tokenizer, role + "_token_id", None) is None:
            value = getattr(tokenizer, fallback + "_token", None)
            if value is None:
                raise ValueError("tokenizer needs a %s or %s token" % (role, fallback))
            setattr(tokenizer, role + "_token", value)
    missing = {role + "_token": "<|%s|>" % role for role in ("pad", "mask")
               if getattr(tokenizer, role + "_token_id") is None}
    roles = _TASK_TOKEN_NAMES
    vocab = tokenizer.get_vocab()
    special = set(tokenizer.all_special_tokens)
    for name in (*missing.values(), *roles.values()):
        if name in vocab and name not in special:
            raise ValueError("special token name is already an ordinary token: %s" % name)
    cast(Callable[..., int], tokenizer.add_special_tokens)(missing)
    cast(Callable[..., int], tokenizer.add_special_tokens)(
        {"additional_special_tokens": list(roles.values())},
        replace_additional_special_tokens=False,
    )
    return NativeTokenSchema(
        vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **{role: tokenizer.convert_tokens_to_ids(name) for role, name in roles.items()},
    )
