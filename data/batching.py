"""Deterministic data batch ordering, rank sharding and progress tracking."""

from __future__ import annotations

import math
from typing import Any, Iterator, Mapping, Optional, cast

import torch
from torch.utils.data import Sampler


def _integer(name: str, value: int, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


class ResumableTrainBatchSampler(Sampler[list[int]]):
    """Shuffle and rank-shard each epoch with an acknowledged batch cursor.

    Each epoch shuffles the global indices with ``seed + epoch``, pads the
    permutation evenly across ranks, selects one strided shard per rank, and
    drops its incomplete final batch. Producing or prefetching indices leaves
    checkpoint progress unchanged; the caller advances it after processing each
    batch. ``len`` remains the full epoch length because Lightning restores its
    own completed-batch count separately.

    Resume requires the same indexed dataset, seed, world size and batch size.
    A provided dataset identity also protects against equal-length data changes.
    """

    def __init__(self, dataset_size: int, *, batch_size: int, seed: int,
                 rank: int, world_size: int,
                 dataset_identity: Optional[str] = None) -> None:
        self.dataset_size = _integer("dataset_size", dataset_size, minimum=1)
        self.batch_size = _integer("batch_size", batch_size, minimum=1)
        self.seed = _integer("seed", seed)
        self.world_size = _integer("world_size", world_size, minimum=1)
        self.rank = _integer("rank", rank)
        if self.rank >= self.world_size:
            raise ValueError("rank must be smaller than world_size")
        if dataset_identity is not None and (not isinstance(dataset_identity, str)
                                              or not dataset_identity):
            raise ValueError("dataset_identity must be a nonempty string or None")
        self.dataset_identity = dataset_identity
        self.drop_last = True
        self.epoch = 0
        self.completed_batches = 0
        if len(self) == 0:
            raise ValueError("training dataset has no complete local batch")

    @property
    def sampler(self) -> ResumableTrainBatchSampler:
        # Lightning finds set_epoch through the public
        # dataloader.batch_sampler.sampler convention.
        return self

    def __len__(self) -> int:
        return math.ceil(self.dataset_size / self.world_size) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        # Snapshot at iterator creation, not later when workers ask for indices.
        epoch, start_batch = self.epoch, self.completed_batches
        generator = torch.Generator().manual_seed(self.seed + epoch)
        indices = cast(list[int], torch.randperm(self.dataset_size, generator=generator).tolist())
        total_size = math.ceil(self.dataset_size / self.world_size) * self.world_size
        padding = total_size - self.dataset_size
        if padding:
            indices.extend((indices * math.ceil(padding / len(indices)))[:padding])
        local = indices[self.rank:total_size:self.world_size]
        return (local[batch * self.batch_size:(batch + 1) * self.batch_size]
                for batch in range(start_batch, len(self)))

    def set_epoch(self, epoch: int) -> None:
        epoch = _integer("epoch", epoch)
        if epoch == self.epoch:
            return
        # A final-batch checkpoint can still carry the preceding epoch in
        # Lightning's loop state. This cursor is already canonical (next, 0).
        if self.completed_batches == 0 and epoch == self.epoch - 1:
            return
        if self.completed_batches:
            raise RuntimeError("cannot change epoch with uncompleted training batches")
        if epoch < self.epoch:
            raise ValueError("cannot rewind the training sampler epoch")
        self.epoch = epoch

    def mark_batch_complete(self) -> None:
        self.completed_batches += 1
        if self.completed_batches == len(self):
            self.epoch += 1
            self.completed_batches = 0

    def state_dict(self) -> dict[str, Any]:
        # Rank is intentionally absent: a rank-0 checkpoint is loaded by every
        # rank. All ranks acknowledge the same number of complete local batches.
        return {"version": 1, "dataset_size": self.dataset_size,
                "batch_size": self.batch_size, "seed": self.seed,
                "world_size": self.world_size, "epoch": self.epoch,
                "completed_batches": self.completed_batches,
                "dataset_identity": self.dataset_identity}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        expected = self.state_dict()
        if set(state_dict) != set(expected):
            raise ValueError("invalid training sampler checkpoint fields")
        for key in ("version", "dataset_size", "batch_size", "seed", "world_size",
                    "dataset_identity"):
            if state_dict[key] != expected[key]:
                raise ValueError(f"training sampler checkpoint differs in {key}")
        epoch = _integer("epoch", state_dict["epoch"])
        completed = _integer("completed_batches", state_dict["completed_batches"])
        if completed >= len(self):
            raise ValueError("completed_batches must use canonical (next epoch, 0) state")
        self.epoch, self.completed_batches = epoch, completed


class WeightedValidationSampler(Sampler[tuple[int, float]]):
    """Equal-length rank shards; repeated fill samples carry zero metric weight.

    Each genuine dataset index has weight one on exactly one rank. Padding only
    balances the number of forward passes; it never adds observations to metrics.
    """

    def __init__(self, dataset_size: int, *, rank: int, world_size: int) -> None:
        self.dataset_size = _integer("dataset_size", dataset_size, minimum=1)
        self.world_size = _integer("world_size", world_size, minimum=1)
        self.rank = _integer("rank", rank)
        if self.rank >= self.world_size:
            raise ValueError("rank must be smaller than world_size")

    def __len__(self) -> int:
        return math.ceil(self.dataset_size / self.world_size)

    def __iter__(self) -> Iterator[tuple[int, float]]:
        for offset in range(len(self)):
            position = self.rank + offset * self.world_size
            yield position % self.dataset_size, float(position < self.dataset_size)
