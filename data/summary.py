"""Summary training examples and prompts for generation evaluation.

Blank or token-empty pairs are omitted. Text encoding to PAD, MASK or task
control tokens is rejected before truncation. Nonblank text is tokenized
without trimming whitespace.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Mapping, cast
from typing import Sequence as TypingSequence

import torch
from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_from_disk
from torch.utils.data import DataLoader, default_collate

from data.types import DataRow, Pathish, TokenBatch

from .tokens import NativeTokenSchema, _validate_tokenizer_schema

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class SummaryConfig:
    max_source_tokens: int
    max_target_tokens: int
    source_field: str = "document"
    reference_field: str = "summary"

    def __post_init__(self) -> None:
        for name, minimum in (("max_source_tokens", 1), ("max_target_tokens", 2)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")

    @property
    def sequence_length(self) -> int:
        return self.max_source_tokens + self.max_target_tokens


def _encode_pairs(sources: TypingSequence[object], references: TypingSequence[object], tokenizer: PreTrainedTokenizerBase) -> Iterable[tuple[list[int], list[int], str, int]]:
    sources = ["" if text is None else str(text) for text in sources]
    references = ["" if text is None else str(text) for text in references]
    pairs = [(source, reference, index) for index, (source, reference) in enumerate(zip(sources, references))
             if source.strip() and reference.strip()]
    if not pairs:
        return ()
    sources, references, indices = map(list, zip(*pairs))
    encoded = tokenizer(
        sources + references, add_special_tokens=False,
        return_attention_mask=False, return_token_type_ids=False,
    )["input_ids"]
    count = len(sources)
    return cast('Iterable[tuple[list[int], list[int], str, int]]', zip(encoded[:count], encoded[count:], references, indices))


def _source_window(source: list[int], pad_id: int, config: SummaryConfig) -> list[int]:
    source = source[:config.max_source_tokens]
    return source + [pad_id] * (config.max_source_tokens - len(source))


def _reject_blocked_tokens(source: TypingSequence[int], reference: TypingSequence[int], forbidden: set[int]) -> None:
    if forbidden.intersection(source) or forbidden.intersection(reference):
        raise ValueError("input text encoded a PAD/MASK/task control token")


def _training_rows(sources: TypingSequence[object], references: TypingSequence[object], *, tokenizer: PreTrainedTokenizerBase, schema: NativeTokenSchema, config: SummaryConfig) -> Iterator[dict[str, Any]]:
    forbidden = set(schema.blocked_token_ids)
    for source, target, _, _index in _encode_pairs(sources, references, tokenizer):
        if not source or not target:
            continue
        _reject_blocked_tokens(source, target, forbidden)
        source = _source_window(source, schema.pad_token_id, config)
        target = [schema.summary_bos] + target + [schema.summary_eos]
        target = target[:config.max_target_tokens]
        target[-1] = schema.summary_eos
        target += [schema.pad_token_id] * (config.max_target_tokens - len(target))
        loss_mask = [0.0] * (config.max_source_tokens + 1)
        loss_mask += [float(token != schema.pad_token_id) for token in target[1:]]
        yield {"input_ids": source + target, "loss_mask": loss_mask}


def _generation_rows(sources: TypingSequence[object], references: TypingSequence[object], *, tokenizer: PreTrainedTokenizerBase, schema: NativeTokenSchema, config: SummaryConfig, indices: TypingSequence[int] | None=None) -> Iterator[dict[str, Any]]:
    forbidden = set(schema.blocked_token_ids)
    for source, target, reference, index in _encode_pairs(sources, references, tokenizer):
        if not source or not target:
            continue
        _reject_blocked_tokens(source, target, forbidden)
        prompt_ids = _source_window(source, schema.pad_token_id, config)
        row = {"prompt_ids": prompt_ids,
               "prompt_text": tokenizer.decode(prompt_ids, skip_special_tokens=True),
               "reference_text": reference}
        if indices is not None:
            row["sample_id"] = indices[index]
        yield row


def summary_training_example(example: Mapping[str, object], tokenizer: PreTrainedTokenizerBase, schema: NativeTokenSchema,
                             config: SummaryConfig) -> TokenBatch | None:
    """Format one pair, or return None for blank or token-empty text.

    The source occupies a fixed prefix. The target starts with summary BOS and
    ends with summary EOS; truncation keeps EOS in the last target slot. Only
    non-PAD target positions after summary BOS receive loss weight.
    """
    _validate_tokenizer_schema(tokenizer, schema)
    row = next(_training_rows(
        [example.get(config.source_field)], [example.get(config.reference_field)],
        tokenizer=tokenizer, schema=schema, config=config), None)
    if row is None:
        return None
    return {"input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "loss_mask": torch.tensor(row["loss_mask"], dtype=torch.float32)}


def summary_generation_example(example: Mapping[str, object], tokenizer: PreTrainedTokenizerBase, schema: NativeTokenSchema,
                               config: SummaryConfig) -> dict[str, Any] | None:
    """Return a right-padded prompt and reference text for a scored example.

    Blank or token-empty pairs return None. Prompts contain only the source;
    generation adds summary BOS and handles stopping at summary EOS.
    """
    _validate_tokenizer_schema(tokenizer, schema)
    row = next(_generation_rows(
        [example.get(config.source_field)], [example.get(config.reference_field)],
        tokenizer=tokenizer, schema=schema, config=config), None)
    if row is None:
        return None
    return {**row, "prompt_ids": torch.tensor(row["prompt_ids"], dtype=torch.long)}


def _encode_batch(sources: TypingSequence[object], references: TypingSequence[object], indices: TypingSequence[int] | None=None, *, tokenizer: PreTrainedTokenizerBase, schema: NativeTokenSchema, config: SummaryConfig, formatter: Callable[..., Iterable[dict[str, Any]]], columns: tuple[str, ...]) -> dict[str, list[Any]]:
    result = {name: [] for name in columns}
    options = {"indices": indices} if indices is not None else {}
    for row in formatter(sources, references, tokenizer=tokenizer, schema=schema, config=config, **options):
        for name in columns:
            result[name].append(row[name])
    return result


def _prepare(dataset: Dataset, tokenizer: PreTrainedTokenizerBase, schema: NativeTokenSchema, config: SummaryConfig, formatter: Callable[..., Iterable[dict[str, Any]]], features: Features, num_workers: int) -> Dataset:
    if type(num_workers) is not int or num_workers < 1:
        raise ValueError("num_workers must be a positive integer")
    _validate_tokenizer_schema(tokenizer, schema)
    if len(dataset) == 0:
        return cast('Dataset', Dataset.from_dict({name: [] for name in features}, features=features).with_format("torch"))
    encode = partial(_encode_batch, tokenizer=tokenizer, schema=schema, config=config,
                     formatter=formatter, columns=tuple(features))
    return cast('Dataset', dataset.map(
        encode, input_columns=[config.source_field, config.reference_field],
        batched=True, remove_columns=dataset.column_names, features=features,
        num_proc=num_workers, with_indices="sample_id" in features,
    ).with_format("torch"))


def prepare_summary_training(dataset: Dataset, tokenizer: PreTrainedTokenizerBase, schema: NativeTokenSchema,
                             config: SummaryConfig, *, num_workers: int = 1) -> Dataset:
    """Batch-tokenize fixed-width input_ids and loss_mask for training or validation."""
    features = _training_features(config)
    return _prepare(dataset, tokenizer, schema, config, _training_rows,
                    features, num_workers)


def _training_features(config: SummaryConfig) -> Features:
    return Features({
        "input_ids": Sequence(Value("int64"), length=config.sequence_length),
        "loss_mask": Sequence(Value("float32"), length=config.sequence_length),
    })


def prepare_summary_splits(
    dataset: Dataset | DatasetDict, tokenizer: PreTrainedTokenizerBase, schema: NativeTokenSchema,
    config: SummaryConfig, *, train_split: str, validation_split: str,
    num_workers: int = 1,
) -> tuple[Dataset, Dataset]:
    """Prepare two named splits, or use a selected Dataset for both outputs."""
    splits = ((dataset[train_split], dataset[validation_split])
              if isinstance(dataset, DatasetDict) else (dataset, dataset))
    return cast(tuple[Dataset, Dataset], tuple(
        prepare_summary_training(split, tokenizer, schema, config,
                                 num_workers=num_workers)
        for split in splits
    ))


def prepare_summary_generation(dataset: Dataset, tokenizer: PreTrainedTokenizerBase, schema: NativeTokenSchema,
                               config: SummaryConfig, *, num_workers: int = 1,
                               include_sample_id: bool = False) -> Dataset:
    """Prepare prompts and references, optionally retaining original split row IDs."""
    features = Features({
        "prompt_ids": Sequence(Value("int64"), length=config.max_source_tokens),
        "prompt_text": Value("string"),
        "reference_text": Value("string"),
    })
    if include_sample_id:
        features["sample_id"] = Value("int64")
    return _prepare(dataset, tokenizer, schema, config, _generation_rows,
                    features, num_workers)


def load_summary_split(path: Pathish, *, split: str) -> Dataset:
    """Read a split from a local saved DatasetDict, or an already selected Dataset."""
    dataset = load_from_disk(str(Path(path).expanduser().resolve(strict=True)))
    return dataset[split] if isinstance(dataset, DatasetDict) else dataset


def collate_summary_generation(examples: list[DataRow]) -> dict[str, Any]:
    """Stack prepared prompts and retain text lists and optional rank weights."""
    batch = {
        "prompt_ids": default_collate([row["prompt_ids"] for row in examples]),
        "prompt_texts": [row["prompt_text"] for row in examples],
        "reference_texts": [row["reference_text"] for row in examples],
    }
    for name in ("sample_weight", "sample_id"):
        if name in examples[0]:
            batch[name] = default_collate([row[name] for row in examples])
    return batch


def make_summary_generation_loader(dataset: Dataset, *, eval_batch_size: int,
                                    num_workers: int, seed: int, rank: int = 0,
                                    world_size: int = 1, pin_memory: bool = False) -> DataLoader[DataRow]:
    """Shard generation rows equally; padding copies carry zero sample_weight."""
    from .loading import make_validation_loader

    return make_validation_loader(
        dataset, eval_batch_size=eval_batch_size, num_workers=num_workers,
        seed=seed, rank=rank, world_size=world_size, pin_memory=pin_memory,
        collate_fn=collate_summary_generation,
    )
