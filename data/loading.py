"""Load prepared datasets with explicit sharding and resumable progress.

Training uses ``Trainer(use_distributed_sampler=False)`` because the loader
already assigns each rank its shard.
"""

from __future__ import annotations

from typing import Any, Iterator, Mapping, Optional, Protocol, cast

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

from data.types import CollateFn, DataRow, MapDataset

from .batching import ResumableTrainBatchSampler, WeightedValidationSampler, _integer


class ResumableTrainDataLoader(DataLoader[DataRow]):
    """Checkpoint only processed batches, independently of worker prefetch.

    Lightning 2.5.5 discovers public ``state_dict/load_state_dict`` methods and
    restores them before creating its CombinedLoader iterator. Recreate the
    loader to restore a checkpoint; restoring an already iterated loader is an
    error. Only one iterator and one acknowledgement owner are supported.
    """

    batch_sampler: ResumableTrainBatchSampler

    def __init__(self, dataset: MapDataset[DataRow], *, batch_sampler: ResumableTrainBatchSampler,
                 num_workers: int, pin_memory: bool = False,
                 collate_fn: Optional[CollateFn] = None) -> None:
        self._iterator_created = False
        self._unacknowledged_batches = 0
        # Isolate worker base-seed creation from the model's global torch RNG.
        generator = torch.Generator().manual_seed(batch_sampler.seed)
        super().__init__(cast(Dataset[DataRow], dataset), batch_sampler=batch_sampler, num_workers=num_workers,
                         pin_memory=pin_memory, collate_fn=collate_fn,
                         persistent_workers=num_workers > 0, generator=generator)

    # The wrapper intentionally exposes Iterator rather than PyTorch's private iterator.
    def __iter__(self) -> Iterator[Any]:  # ty: ignore[invalid-method-override]
        if self._unacknowledged_batches:
            raise RuntimeError("cannot restart a loader with an unacknowledged batch")
        self._iterator_created = True
        iterator = super().__iter__()  # Worker prefetch starts here.
        return self._delivered_batches(iterator)

    def _delivered_batches(self, iterator: Iterator[Any]) -> Iterator[Any]:
        for batch in iterator:
            self._unacknowledged_batches += 1
            yield batch

    def mark_batch_complete(self) -> None:
        if self._unacknowledged_batches == 0:
            raise RuntimeError("no delivered training batch to acknowledge")
        self._unacknowledged_batches -= 1
        self.batch_sampler.mark_batch_complete()

    def state_dict(self) -> dict[str, Any]:
        return self.batch_sampler.state_dict()

    @property
    def has_pending_batch(self) -> bool:
        """Whether a delivered batch has not reached its completion callback."""
        return self._unacknowledged_batches != 0

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if self._iterator_created:
            raise RuntimeError("restore training loader state before creating any iterator")
        self.batch_sampler.load_state_dict(state_dict)


class _WeightedValidationDataset(Dataset[dict[str, Any]]):
    def __init__(self, dataset: MapDataset[DataRow]) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    # The validation sampler supplies (index, weight) pairs by design.
    def __getitem__(self, item: tuple[int, float]) -> dict[str, Any]:  # ty: ignore[invalid-method-override]
        index, weight = item
        row = dict(self.dataset[index])
        if "sample_weight" in row:
            raise ValueError("sample_weight is reserved for validation sharding")
        row["sample_weight"] = torch.tensor(weight, dtype=torch.float32)
        return row


def make_train_loader(dataset: MapDataset[DataRow], *, local_batch_size: int, num_workers: int,
                      seed: int, rank: int, world_size: int, pin_memory: bool = False,
                      collate_fn: Optional[CollateFn] = None,
                      dataset_identity: Optional[str] = None) -> ResumableTrainDataLoader:
    if dataset_identity is None:
        dataset_identity = getattr(dataset, "dataset_identity", None)
    sampler = ResumableTrainBatchSampler(len(dataset), batch_size=local_batch_size,
                                         seed=seed, rank=rank, world_size=world_size,
                                         dataset_identity=dataset_identity)
    return ResumableTrainDataLoader(dataset, batch_sampler=sampler,
                                    num_workers=num_workers, pin_memory=pin_memory,
                                    collate_fn=collate_fn)


def make_validation_loader(dataset: MapDataset[DataRow], *, eval_batch_size: int, num_workers: int,
                           seed: int, rank: int, world_size: int,
                           pin_memory: bool = False,
                           collate_fn: Optional[CollateFn] = None) -> DataLoader[DataRow]:
    _integer("seed", seed)
    sampler = WeightedValidationSampler(len(dataset), rank=rank, world_size=world_size)
    return DataLoader(_WeightedValidationDataset(dataset), batch_size=eval_batch_size,
                      sampler=sampler, num_workers=num_workers, pin_memory=pin_memory,
                      collate_fn=collate_fn, persistent_workers=num_workers > 0, drop_last=False,
                      generator=torch.Generator().manual_seed(seed))


class TrainingDataModule(L.LightningDataModule):
    """Deterministic map-style datasets, with dimensions explicit at construction.

    Datasets must be deterministic: worker-side stochastic augmentation state is
    not checkpointed. Collators may construct task-specific batches from rows.
    Rank and world size come from the attached Trainer after strategy setup;
    standalone callers may supply both explicitly. Restore with a new instance.
    ``dataset_identity`` overrides the dataset's ``dataset_identity`` attribute.
    Without either identity (e.g. a synthetic Dataset), resume checks length only.
    """

    def __init__(self, train_dataset: Optional[MapDataset[DataRow]], val_dataset: MapDataset[DataRow], *,
                 local_batch_size: int, eval_batch_size: int, num_workers: int,
                 seed: int, pin_memory: bool = False,
                 rank: Optional[int] = None, world_size: Optional[int] = None,
                 dataset_identity: Optional[str] = None,
                 train_collate_fn: Optional[CollateFn] = None,
                 val_collate_fn: Optional[CollateFn] = None) -> None:
        super().__init__()
        self.train_dataset, self.val_dataset = train_dataset, val_dataset
        self.local_batch_size = local_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.pin_memory = pin_memory
        self.dataset_identity = dataset_identity
        self.train_collate_fn = train_collate_fn
        self.val_collate_fn = val_collate_fn
        if (rank is None) != (world_size is None):
            raise ValueError("supply both rank and world_size, or neither")
        self.rank, self.world_size = rank, world_size
        self._train_loader: Optional[ResumableTrainDataLoader] = None
        self._val_loader: Optional[DataLoader[DataRow]] = None

    def _shard(self) -> tuple[int, int]:
        if self.trainer is not None:
            rank, world_size = self.trainer.global_rank, self.trainer.world_size
            if self.rank is not None and (self.rank, self.world_size) != (rank, world_size):
                raise ValueError("explicit data shard differs from the attached Trainer")
            return rank, world_size
        if self.rank is None:
            raise RuntimeError("attach a Trainer or explicitly supply rank and world_size")
        return self.rank, cast(int, self.world_size)

    def train_dataloader(self) -> ResumableTrainDataLoader:
        if self._train_loader is None:
            rank, world_size = self._shard()
            self._train_loader = make_train_loader(
                cast(MapDataset[DataRow], self.train_dataset), local_batch_size=self.local_batch_size,
                num_workers=self.num_workers, seed=self.seed, rank=rank,
                world_size=world_size, pin_memory=self.pin_memory,
                collate_fn=self.train_collate_fn,
                dataset_identity=self.dataset_identity)
        return self._train_loader

    def val_dataloader(self) -> DataLoader[DataRow]:
        if self._val_loader is None:
            rank, world_size = self._shard()
            self._val_loader = make_validation_loader(
                self.val_dataset, eval_batch_size=self.eval_batch_size,
                num_workers=self.num_workers, seed=self.seed, rank=rank,
                world_size=world_size, pin_memory=self.pin_memory,
                collate_fn=self.val_collate_fn)
        return self._val_loader

    def mark_train_batch_complete(self) -> None:
        if self._train_loader is None:
            raise RuntimeError("training loader has not been created")
        self._train_loader.mark_batch_complete()


class _TrainerData(Protocol):
    """Fields installed by Lightning's data and callback connectors."""

    datamodule: TrainingDataModule
    callbacks: list[L.Callback]


class TrainingDataProgress(L.Callback):
    """Install before ModelCheckpoint; do not also acknowledge in the module.

    Lightning calls callback batch-end hooks before the module's batch-end hook.
    This ordering ensures a ModelCheckpoint sees the just-completed batch cursor.
    Checkpoints must be taken at completed optimizer updates, not partial
    accumulation windows, for exact optimizer resume.
    """

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        runtime = cast(_TrainerData, trainer)
        if not isinstance(runtime.datamodule, TrainingDataModule):
            raise TypeError("TrainingDataProgress requires TrainingDataModule")
        if trainer.train_dataloader is not runtime.datamodule.train_dataloader():
            raise RuntimeError("training loader was replaced; use_distributed_sampler must be False")
        if sum(isinstance(cb, TrainingDataProgress) for cb in runtime.callbacks) != 1:
            raise RuntimeError("install exactly one TrainingDataProgress callback")
        position = runtime.callbacks.index(self)
        if any(isinstance(cb, L.pytorch.callbacks.ModelCheckpoint)
               for cb in runtime.callbacks[:position]):
            raise RuntimeError("TrainingDataProgress must precede ModelCheckpoint")

    def on_train_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule,
                           outputs: Any, batch: Any, batch_idx: int) -> None:
        cast(_TrainerData, trainer).datamodule.mark_train_batch_complete()
