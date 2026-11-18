"""Prepare datasets before distributed launch and assemble their data loaders."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader

from data.types import DataRow, Pathish

from .loading import TrainingDataModule
from .pretrain import PackingConfig, select_corpus_split
from .pretrain_storage import TokenDataset, build_token_store
from .summary import (
    SummaryConfig,
    load_summary_split,
    make_summary_generation_loader,
    prepare_summary_generation,
    prepare_summary_splits,
)
from .summary_storage import (
    load_summary_store,
    prepare_summary_store,
    summary_store_metadata,
)
from .tokens import NativeTokenSchema

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


def prepare_pretraining_datasets(
    corpus_path: Pathish,
    tokenizer: PreTrainedTokenizerBase,
    schema: NativeTokenSchema,
    *,
    profile: str,
    tokenizer_id: str,
    train_store_dir: Pathish,
    validation_store_dir: Pathish,
    train_packing: PackingConfig,
    validation_packing: PackingConfig,
    column: str = "text",
) -> tuple[TokenDataset, TokenDataset]:
    """Select, pack and build or reopen the two profile-defined token stores."""
    corpus_path = Path(corpus_path).expanduser().resolve(strict=True)
    dataset = load_from_disk(str(corpus_path))
    requests = (
        ("train", train_store_dir, train_packing),
        ("validation", validation_store_dir, validation_packing),
    )
    corpora = [
        select_corpus_split(
            dataset, profile=profile, split=split,
            column=column,
        )
        for split, _, _ in requests
    ]
    for corpus, (split, _, packing) in zip(corpora, requests):
        if packing.preprocessing != corpus.preprocessing:
            raise ValueError(
                f"{split} packing preprocessing must be {corpus.preprocessing!r} "
                f"for profile {profile!r}"
            )
    prepared = []
    for corpus, (_, store_dir, packing) in zip(corpora, requests):
        prepared.append(build_token_store(
            corpus, tokenizer, schema, packing,
            store_dir=store_dir, source=corpus.source,
            split=corpus.split, tokenizer_id=tokenizer_id,
        ))
    return prepared[0], prepared[1]


def prepare_pretraining_datamodule(
    corpus_path: Pathish,
    tokenizer: PreTrainedTokenizerBase,
    schema: NativeTokenSchema,
    *,
    profile: str,
    tokenizer_id: str,
    train_store_dir: Pathish,
    validation_store_dir: Pathish,
    train_packing: PackingConfig,
    validation_packing: PackingConfig,
    local_batch_size: int,
    eval_batch_size: int,
    loader_num_workers: int,
    seed: int,
    column: str = "text",
    pin_memory: bool = False,
    rank: int | None = None,
    world_size: int | None = None,
) -> TrainingDataModule:
    """Prepare token stores and assemble loaders with loader_num_workers workers."""
    train_dataset, validation_dataset = prepare_pretraining_datasets(
        corpus_path, tokenizer, schema, profile=profile, tokenizer_id=tokenizer_id,
        train_store_dir=train_store_dir,
        validation_store_dir=validation_store_dir,
        train_packing=train_packing, validation_packing=validation_packing,
        column=column,
    )
    return TrainingDataModule(
        train_dataset, validation_dataset,
        local_batch_size=local_batch_size, eval_batch_size=eval_batch_size,
        num_workers=loader_num_workers, seed=seed, pin_memory=pin_memory,
        rank=rank, world_size=world_size,
    )


def prepare_summary_datasets(
    dataset_path: Pathish,
    tokenizer: PreTrainedTokenizerBase,
    schema: NativeTokenSchema,
    config: SummaryConfig,
    *,
    train_split: str,
    validation_split: str,
    preparation_num_workers: int,
) -> tuple[Dataset, Dataset]:
    """Load raw summary data and prepare train/validation rows."""
    source = load_from_disk(str(Path(dataset_path).expanduser().resolve(strict=True)))
    return prepare_summary_splits(
        source, tokenizer, schema, config, train_split=train_split,
        validation_split=validation_split, num_workers=preparation_num_workers,
    )


def prepare_summary_datamodule(
    dataset_path: Pathish,
    tokenizer: PreTrainedTokenizerBase,
    schema: NativeTokenSchema,
    config: SummaryConfig,
    *,
    train_split: str,
    validation_split: str,
    preparation_num_workers: int,
    local_batch_size: int,
    eval_batch_size: int,
    loader_num_workers: int,
    seed: int,
    pin_memory: bool = False,
    rank: int | None = None,
    world_size: int | None = None,
    dataset_identity: str | None = None,
) -> TrainingDataModule:
    """Prepare summary rows with preparation_num_workers, then assemble loaders."""
    train_dataset, validation_dataset = prepare_summary_datasets(
        dataset_path, tokenizer, schema, config,
        train_split=train_split, validation_split=validation_split,
        preparation_num_workers=preparation_num_workers,
    )
    return TrainingDataModule(
        train_dataset, validation_dataset,
        local_batch_size=local_batch_size, eval_batch_size=eval_batch_size,
        num_workers=loader_num_workers, seed=seed, pin_memory=pin_memory,
        rank=rank, world_size=world_size, dataset_identity=dataset_identity,
    )


def prepare_summary_generation_dataset(
    dataset_path: Pathish,
    tokenizer: PreTrainedTokenizerBase,
    schema: NativeTokenSchema,
    config: SummaryConfig,
    *,
    split: str,
    preparation_num_workers: int,
) -> Dataset:
    """Load one explicit split and prepare its summary generation rows."""
    dataset = load_summary_split(dataset_path, split=split)
    return prepare_summary_generation(
        dataset, tokenizer, schema, config,
        num_workers=preparation_num_workers,
    )


def prepare_summary_generation_loader(
    dataset_path: Pathish,
    tokenizer: PreTrainedTokenizerBase,
    schema: NativeTokenSchema,
    config: SummaryConfig,
    *,
    split: str,
    preparation_num_workers: int,
    eval_batch_size: int,
    loader_num_workers: int,
    seed: int,
    rank: int = 0,
    world_size: int = 1,
    pin_memory: bool = False,
) -> DataLoader[DataRow]:
    """Prepare one summary split and assemble its generation loader."""
    dataset = prepare_summary_generation_dataset(
        dataset_path, tokenizer, schema, config, split=split,
        preparation_num_workers=preparation_num_workers,
    )
    return make_summary_generation_loader(
        dataset, eval_batch_size=eval_batch_size,
        num_workers=loader_num_workers, seed=seed, rank=rank,
        world_size=world_size, pin_memory=pin_memory,
    )


__all__ = [
    "summary_store_metadata",
    "prepare_summary_store",
    "load_summary_store",
    "prepare_pretraining_datasets",
    "prepare_pretraining_datamodule",
    "prepare_summary_datasets",
    "prepare_summary_datamodule",
    "prepare_summary_generation_dataset",
    "prepare_summary_generation_loader",
]
