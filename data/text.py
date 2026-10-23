"""Packed ordinary-text prompts and references for generation evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from torch.utils.data import DataLoader, Dataset, default_collate

from data.types import DataRow, MapDataset, Pathish

from .pretrain import PackingConfig, load_corpus_split
from .pretrain_storage import TokenDataset, build_token_store
from .tokens import NativeTokenSchema

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


def _nonnegative_integer(name: str, value: int) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _positive_integer(name: str, value: int) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


class TextGenerationDataset(Dataset[dict[str, Any]]):
    """A selected view of packed token windows with decoded score references."""

    def __init__(
        self,
        store: TokenDataset,
        tokenizer: PreTrainedTokenizerBase,
        indices: list[int],
        *,
        prompt_length: int,
        generation_length: int,
        source_count: int,
    ) -> None:
        self.store = store
        self.tokenizer = tokenizer
        self.indices = tuple(indices)
        self.prompt_length = prompt_length
        self.generation_length = generation_length
        self.source_count = source_count
        self.eligible_count = len(store)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        store_index = self.indices[index]
        token_ids = self.store[store_index]["input_ids"]
        prompt_ids = token_ids[:self.prompt_length]
        reference_ids = token_ids[self.prompt_length:]
        decode = self.tokenizer.decode
        return {
            "prompt_ids": prompt_ids,
            "prompt_text": decode(prompt_ids.tolist(), skip_special_tokens=True),
            "reference_text": decode(reference_ids.tolist(), skip_special_tokens=True),
            "reference_full_text": decode(token_ids.tolist(), skip_special_tokens=True),
            "sample_id": store_index,
        }


def prepare_text_generation(
    corpus_path: Pathish,
    tokenizer: PreTrainedTokenizerBase,
    schema: NativeTokenSchema,
    *,
    profile: str,
    revision: str,
    tokenizer_id: str,
    store_dir: Pathish,
    prompt_length: int,
    generation_length: int,
    num_samples: int,
    shuffle: bool = False,
    seed: int = 0,
    pack_batch_size: int = 1000,
    num_workers: int = 1,
    column: str = "text",
    document_eos: bool = True,
    window_bos_eos: bool = True,
) -> TextGenerationDataset:
    """Prepare a fixed-size view of packed validation windows.

    ``prompt_length + generation_length`` is the packed window width. With an
    empty prompt, the generation length is the complete window width, including
    a window BOS when ``window_bos_eos`` is enabled. Selection is either the
    first ``num_samples`` store rows or a seeded permutation of all eligible
    rows. Stable sample IDs are the selected store-row indices.
    """
    prompt_length = _nonnegative_integer("prompt_length", prompt_length)
    generation_length = _positive_integer("generation_length", generation_length)
    num_samples = _positive_integer("num_samples", num_samples)
    seed = _nonnegative_integer("seed", seed)
    if type(shuffle) is not bool:
        raise TypeError("shuffle must be a bool")

    corpus = load_corpus_split(
        corpus_path, profile=profile, split="validation", revision=revision,
        column=column,
    )
    source_count = len(corpus)
    packing = PackingConfig(
        sequence_length=prompt_length + generation_length,
        pack_batch_size=pack_batch_size,
        num_workers=num_workers,
        preprocessing=corpus.preprocessing,
        document_eos=document_eos,
        window_bos_eos=window_bos_eos,
    )
    store = build_token_store(
        corpus, tokenizer, schema, packing,
        store_dir=Path(store_dir), source=corpus.source,
        revision=corpus.revision, split=corpus.split,
        tokenizer_id=tokenizer_id,
    )
    eligible_count = len(store)
    if eligible_count < num_samples:
        raise ValueError(
            f"requested {num_samples} text samples but only "
            f"{eligible_count} packed windows are eligible"
        )
    if shuffle:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(eligible_count, generator=generator)[:num_samples].tolist()
    else:
        indices = list(range(num_samples))
    return TextGenerationDataset(
        store, tokenizer, indices, prompt_length=prompt_length,
        generation_length=generation_length, source_count=source_count,
    )


def _collate_text_generation(examples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "prompt_ids": default_collate([row["prompt_ids"] for row in examples]),
        "prompt_texts": [row["prompt_text"] for row in examples],
        "reference_texts": [row["reference_text"] for row in examples],
        "reference_full_texts": [row["reference_full_text"] for row in examples],
        "sample_id": default_collate([row["sample_id"] for row in examples]),
    }


def make_text_generation_loader(
    dataset: MapDataset[DataRow],
    *,
    eval_batch_size: int,
    num_workers: int,
    seed: int,
    pin_memory: bool = False,
) -> DataLoader[DataRow]:
    """Batch a selected text-generation dataset without dropping its tail."""
    eval_batch_size = _positive_integer("eval_batch_size", eval_batch_size)
    num_workers = _nonnegative_integer("num_workers", num_workers)
    seed = _nonnegative_integer("seed", seed)
    return DataLoader(
        cast(Dataset[DataRow], dataset),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_text_generation,
        persistent_workers=num_workers > 0,
        drop_last=False,
        generator=torch.Generator().manual_seed(seed),
    )


__all__ = [
    "TextGenerationDataset", "make_text_generation_loader",
    "prepare_text_generation",
]
