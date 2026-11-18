"""Persistent prepared datasets for summary training."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, cast

from datasets import Dataset, DatasetDict, Features, load_from_disk

from data.types import Metadata, Pathish

from .summary import SummaryConfig, _training_features, prepare_summary_splits
from .tokens import NativeTokenSchema

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


_MANIFEST = "manifest.json"
_SPLITS = ("train", "validation")


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def summary_store_metadata(
    schema: NativeTokenSchema,
    config: SummaryConfig,
    *,
    profile: str,
    tokenizer_id: str,
    train_split: str,
    validation_split: str,
) -> Metadata:
    """Describe the inputs that determine prepared summary rows."""
    if train_split == validation_split:
        raise ValueError("train_split and validation_split must be distinct")
    return {
        "format": "summary-arrow",
        "task": {
            "kind": "summary",
            "profile": profile,
            "train_split": train_split,
            "validation_split": validation_split,
            "summary": asdict(config),
        },
        "tokenizer": tokenizer_id,
        "special_ids": schema.to_dict(),
    }


def _manifest(path: Path, expected_metadata: Mapping[str, Any]) -> tuple[dict[str, int], str]:
    manifest = json.loads((path / _MANIFEST).read_text(encoding="utf-8"))
    metadata = manifest["metadata"]
    counts = manifest["num_rows"]
    if metadata != expected_metadata:
        raise ValueError("summary store metadata mismatch")
    if (set(counts) != set(_SPLITS)
            or any(type(counts[name]) is not int or counts[name] < 1
                   for name in _SPLITS)):
        raise ValueError("invalid summary store row counts")
    identity = _json({
        "metadata": metadata, "num_rows": counts,
    })
    return cast('tuple[dict[str, int], str]', (counts, identity))


def _load_split(path: Path, split: str, rows: int, identity: str,
                expected_features: Features) -> Dataset:
    dataset = load_from_disk(str(path / split))
    if not isinstance(dataset, Dataset) or dataset.features != expected_features:
        raise ValueError(f"prepared summary {split} schema does not match manifest")
    if len(dataset) != rows:
        raise ValueError(f"prepared summary {split} row count does not match manifest")
    dataset = dataset.with_format("torch")
    dataset.dataset_identity = identity
    return cast('Dataset', dataset)


def load_summary_store(
    store_dir: Pathish,
    *,
    expected_metadata: Mapping[str, Any],
    validation_only: bool = False,
) -> tuple[Dataset | None, Dataset]:
    """Open a prepared store, optionally without opening its training split."""
    path = Path(store_dir).expanduser().resolve()
    counts, identity = _manifest(path, expected_metadata)
    expected_features = _training_features(SummaryConfig(
        **expected_metadata["task"]["summary"]
    ))
    train = None if validation_only else _load_split(
        path, "train", counts["train"], identity, expected_features,
    )
    validation = _load_split(
        path, "validation", counts["validation"], identity, expected_features,
    )
    return train, validation


def prepare_summary_store(
    dataset_path: Pathish,
    tokenizer: PreTrainedTokenizerBase,
    schema: NativeTokenSchema,
    config: SummaryConfig,
    *,
    store_dir: Pathish,
    profile: str,
    tokenizer_id: str,
    train_split: str,
    validation_split: str,
    preparation_num_workers: int,
) -> tuple[Dataset, Dataset]:
    """Build a prepared summary store once, or reopen the complete store."""
    metadata = summary_store_metadata(
        schema, config, profile=profile, tokenizer_id=tokenizer_id,
        train_split=train_split,
        validation_split=validation_split,
    )
    target = Path(store_dir).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.parent / (target.name + ".lock")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if target.exists():
            return cast(tuple[Dataset, Dataset], load_summary_store(target, expected_metadata=metadata))

        source = load_from_disk(str(
            Path(dataset_path).expanduser().resolve(strict=True)
        ))
        if not isinstance(source, DatasetDict):
            raise TypeError("summary store preparation requires a DatasetDict")
        missing = [name for name in (train_split, validation_split)
                   if name not in source]
        if missing:
            raise ValueError("raw summary DatasetDict is missing split(s): "
                             + ", ".join(repr(name) for name in missing))

        prepared = dict(zip(_SPLITS, prepare_summary_splits(
            source, tokenizer, schema, config, train_split=train_split,
            validation_split=validation_split, num_workers=preparation_num_workers,
        )))
        counts = {role: len(dataset) for role, dataset in prepared.items()}
        for role, count in counts.items():
            if count == 0:
                raise ValueError(
                    f"prepared summary {role} split must contain at least one row"
                )

        with tempfile.TemporaryDirectory(
            prefix=f".{target.name}.building-", dir=str(target.parent),
        ) as directory:
            temporary = Path(directory)
            for role, dataset in prepared.items():
                dataset.save_to_disk(str(temporary / role))
            (temporary / _MANIFEST).write_text(
                _json({"metadata": metadata, "num_rows": counts}),
                encoding="utf-8",
            )
            if target.exists():
                raise FileExistsError(
                    f"summary store destination appeared during build: {target}"
                )
            os.rename(temporary, target)
    return cast(tuple[Dataset, Dataset], load_summary_store(target, expected_metadata=metadata))
