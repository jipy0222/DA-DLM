"""Pretraining corpus selection, text encoding and fixed-window packing."""

from __future__ import annotations

import multiprocessing
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator, List, cast

from datasets import Dataset, DatasetDict, load_from_disk

from data.types import Metadata, Pathish

from .tokens import NativeTokenSchema, _validate_tokenizer_schema

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class CorpusSplit:
    rows: Dataset
    source: Metadata
    split: str
    preprocessing: str
    column: str = "text"

    def __iter__(self) -> Iterator[str]:
        return cast('Iterator[str]', (row[self.column] for row in self.rows))

    def __len__(self) -> int:
        return len(self.rows)


def select_corpus_split(dataset: Dataset | DatasetDict, *, profile: str, split: str,
                        column: str = "text") -> CorpusSplit:
    """Select profile-defined raw rows before tokenization and packing.

    ``owt``: train excludes the final 100000 rows; validation is those rows.
    ``owt_tiny``: first 100000 train rows, final 1500 validation rows.
    ``lm1b``: the provided train/test splits; test serves as validation, not
    as an independent final test set.
    """
    if not isinstance(dataset, (Dataset, DatasetDict)):
        raise TypeError("dataset must be a Hugging Face Dataset or DatasetDict")
    if split not in {"train", "validation"}:
        raise ValueError("split must be train or validation")
    if profile == "lm1b":
        if not isinstance(dataset, DatasetDict):
            raise TypeError("lm1b requires a DatasetDict with train and test splits")
        rows = dataset["train" if split == "train" else "test"]
        preprocessing = "lm1b"
    elif profile in ("owt", "owt_tiny"):
        rows = dataset["train"] if isinstance(dataset, DatasetDict) else dataset
        size = len(rows)
        minimum = 100001 if profile == "owt" else 101500
        if size < minimum:
            raise ValueError("raw %s snapshot is too small for its nonoverlapping splits" % profile)
        if profile == "owt":
            start, stop = (0, size - 100000) if split == "train" else (size - 100000, size)
        else:
            start, stop = (0, 100000) if split == "train" else (size - 1500, size)
        rows = rows.select(range(start, stop))
        preprocessing = "none"
    else:
        raise ValueError("profile must be owt, owt_tiny or lm1b")
    identity = {"column": column}
    return CorpusSplit(rows, identity, profile + ":" + split, preprocessing, column)


def load_corpus_split(path: Pathish, *, profile: str, split: str,
                      column: str = "text") -> CorpusSplit:
    """Load a local saved snapshot and select its training or validation rows."""
    path = Path(path).expanduser().resolve(strict=True)
    dataset = load_from_disk(str(path))
    return select_corpus_split(dataset, profile=profile, split=split,
                               column=column)


@dataclass(frozen=True)
class PackingConfig:
    sequence_length: int
    pack_batch_size: int = 1000
    num_workers: int = 1
    preprocessing: str = "none"
    document_eos: bool = True
    window_bos_eos: bool = True

    def __post_init__(self) -> None:
        for name in ("sequence_length", "pack_batch_size", "num_workers"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("%s must be an integer" % name)
        for name in ("document_eos", "window_bos_eos"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError("%s must be a boolean" % name)
        minimum_length = 3 if self.window_bos_eos else 1
        if self.sequence_length < minimum_length:
            if self.window_bos_eos:
                raise ValueError("sequence_length must reserve BOS, content and EOS")
            raise ValueError("sequence_length must be positive")
        if self.pack_batch_size < 1 or self.num_workers < 1:
            raise ValueError("pack_batch_size and num_workers must be positive")
        if self.preprocessing not in {"none", "lm1b"}:
            raise ValueError("preprocessing must be 'none' or 'lm1b'")

    def to_metadata(self) -> Metadata:
        # CPU concurrency is execution policy, not a source of shard boundaries.
        return {
            "layout": "concatenated_windows",
            "sequence_length": self.sequence_length,
            "pack_batch_size": self.pack_batch_size,
            "document_eos": self.document_eos,
            "window_bos_eos": self.window_bos_eos,
            "remainder": "drop_per_pack_batch",
            "preprocessing": self.preprocessing,
        }


def lm1b_detokenize(text: str) -> str:
    """Remove LM1B's extra spacing around punctuation, quotes and URLs."""
    text = text.replace('http : / / ', 'http://')
    text = text.replace('https : / / ', 'https://')
    text = re.sub(r" '(\w+)", r"'\1", text)
    text = re.sub(r' (\w+) \. ', r' \1. ', text)
    text = re.sub(r' (\w+) \.$', r' \1.', text)
    text = text.replace(' ? ', '? ')
    text = re.sub(r' \?$', '?', text)
    text = text.replace(' ! ', '! ')
    text = re.sub(r' \!$', '!', text)
    text = text.replace(' , ', ', ')
    text = text.replace(' : ', ': ')
    text = text.replace(' ; ', '; ')
    text = text.replace(' / ', '/')
    text = re.sub(r'\" ([^\"]+) \"', r'"\1"', text)
    text = re.sub(r"' ([^']+) '", r"'\1'", text)
    text = re.sub(r'\( ([^\(\)]+) \)', r'(\1)', text)
    text = re.sub(r'\[ ([^\[\]]+) \]', r'[\1]', text)
    text = text.replace('$ ', '$')
    text = text.replace('£ ', '£')
    return text


def _encode(text: str, tokenizer: PreTrainedTokenizerBase, preprocessing: str) -> List[int]:
    if not isinstance(text, str):
        raise TypeError("text packing inputs must contain strings")
    if preprocessing == "lm1b":
        text = lm1b_detokenize(text)
    return list(tokenizer.encode(text, add_special_tokens=False))


_WORKER_TOKENIZER = None
_WORKER_PREPROCESSING = None


def _initialize_encoder(tokenizer: PreTrainedTokenizerBase, preprocessing: str) -> None:
    global _WORKER_TOKENIZER, _WORKER_PREPROCESSING
    _WORKER_TOKENIZER = tokenizer
    _WORKER_PREPROCESSING = preprocessing


def _encode_worker(text: str) -> List[int]:
    return _encode(text, cast("PreTrainedTokenizerBase", _WORKER_TOKENIZER),
                   cast(str, _WORKER_PREPROCESSING))


def iter_token_windows(
    texts: Iterable[str], tokenizer: PreTrainedTokenizerBase, schema: NativeTokenSchema,
    config: PackingConfig,
) -> Iterator[List[int]]:
    """Encode ordered documents and emit full, non-PAD windows.

    Each fixed batch of ``pack_batch_size`` source rows is packed separately.
    Its final remainder is discarded, including for the final short batch.
    Worker count never changes those boundaries. Memory is bounded by one such
    source batch and its encoded documents, not the complete corpus.
    """
    _validate_tokenizer_schema(tokenizer, schema)
    if isinstance(texts, str):
        raise TypeError("supply an iterable of documents, not one string")
    source = iter(texts)
    width = config.sequence_length - (2 if config.window_bos_eos else 0)
    forbidden = set(schema.blocked_token_ids)
    executor = None
    try:
        if config.num_workers > 1:
            executor = ProcessPoolExecutor(
                max_workers=config.num_workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_encoder,
                initargs=(tokenizer, config.preprocessing),
            )
        while True:
            batch = list(islice(source, config.pack_batch_size))
            if not batch:
                break
            if executor is None:
                encoded = (_encode(text, tokenizer, config.preprocessing) for text in batch)
            else:
                encoded = executor.map(_encode_worker, batch, chunksize=max(1, len(batch) // (4 * config.num_workers)))
            pending = []
            for document in encoded:
                if forbidden.intersection(document):
                    raise ValueError("input text encoded a PAD/MASK/task control token")
                pending.extend(document)
                if config.document_eos:
                    pending.append(schema.eos_token_id)
                complete_length = len(pending) // width * width
                for start in range(0, complete_length, width):
                    window = pending[start:start + width]
                    if config.window_bos_eos:
                        window = [schema.bos_token_id] + window + [schema.eos_token_id]
                    yield window
                pending = pending[complete_length:]
            # No carry into the next fixed source batch.
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
