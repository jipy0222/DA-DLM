"""Local tokenizer loading and native vocabulary compatibility."""

from __future__ import annotations

from typing import cast

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from data.tokens import NativeTokenSchema, prepare_tokenizer
from data.types import Pathish


def load_tokenizer(path: Pathish) -> PreTrainedTokenizerBase:
    return cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False))


def check_vocabulary(tokenizer: PreTrainedTokenizerBase, tokens: NativeTokenSchema,
                      other: PreTrainedTokenizerBase) -> None:
    if tokens != prepare_tokenizer(other):
        raise ValueError("tokenizer special-token roles differ")
    if tokenizer.get_vocab() != other.get_vocab():
        raise ValueError("tokenizer vocabulary differs; token IDs must identify the same tokens")
