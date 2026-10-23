"""Restricted checkpoint adaptation for changed distributed training layouts."""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Optional, cast

from lightning.fabric.plugins.io import CheckpointIO, TorchCheckpointIO
from lightning.pytorch import Callback, LightningModule, Trainer
from torch.serialization import MAP_LOCATION
from torch.types import Storage

from training._types import Pathish

_METADATA_KEY = "dadlm_resume_layout"
_METADATA_VERSION = 2
_SAMPLER_FIELDS = {
    "version", "dataset_size", "batch_size", "seed", "world_size",
    "epoch", "completed_batches", "dataset_identity",
}
_PROGRESS_FIELDS = {"ready", "completed", "started", "processed"}


def _positive_integer(name: str, value: int) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


@dataclass(frozen=True)
class ResumeLayout:
    """Target layout for a restricted full-state training resume."""

    local_batch_size: int
    world_size: int
    accumulation_steps: int

    def __post_init__(self) -> None:
        _positive_integer("local_batch_size", self.local_batch_size)
        _positive_integer("world_size", self.world_size)
        _positive_integer("accumulation_steps", self.accumulation_steps)

    @property
    def global_microbatch_size(self) -> int:
        return self.local_batch_size * self.world_size

    @property
    def global_batch_size(self) -> int:
        return self.global_microbatch_size * self.accumulation_steps


@dataclass(frozen=True)
class ResumeRuntime:
    """Training-math settings that cannot change during a native resume."""

    precision: str
    gradient_clip_val: Optional[float]
    gradient_clip_algorithm: str | None

    def __post_init__(self) -> None:
        precision = getattr(self.precision, "value", self.precision)
        algorithm = getattr(self.gradient_clip_algorithm, "value",
                            self.gradient_clip_algorithm)
        if not isinstance(precision, str) or not precision:
            raise ValueError("precision must be a nonempty string")
        clip = self.gradient_clip_val
        if clip is not None:
            if isinstance(clip, bool) or not isinstance(clip, (int, float)):
                raise TypeError("gradient_clip_val must be a number or None")
            clip = float(clip)
            if not math.isfinite(clip) or clip < 0:
                raise ValueError("gradient_clip_val must be finite and nonnegative")
            if algorithm is None:
                algorithm = "norm"
            if not isinstance(algorithm, str) or not algorithm:
                raise ValueError("gradient_clip_algorithm must be a nonempty string")
        else:
            algorithm = "none"
        object.__setattr__(self, "precision", precision)
        object.__setattr__(self, "gradient_clip_val", clip)
        object.__setattr__(self, "gradient_clip_algorithm", algorithm)


@dataclass(frozen=True)
class _SavedResumeMetadata:
    accumulation_steps: int
    runtime: ResumeRuntime


class ResumeMetadata(Callback):
    """Record the layout and runtime settings needed to validate a later resume."""

    def on_save_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        del pl_module
        runtime = ResumeRuntime(
            trainer.precision,
            trainer.gradient_clip_val,
            trainer.gradient_clip_algorithm,
        )
        checkpoint[_METADATA_KEY] = {
            "version": _METADATA_VERSION,
            "accumulation_steps": _positive_integer(
                "accumulate_grad_batches", trainer.accumulate_grad_batches
            ),
            **asdict(runtime),
        }


def _mapping(name: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"invalid Lightning 2.5.5 checkpoint field: {name}")
    return value


def _counter(name: str, value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"invalid non-negative checkpoint counter: {name}")
    return value


def _saved_metadata(checkpoint: Mapping[str, Any]) -> Optional[_SavedResumeMetadata]:
    metadata = checkpoint.get(_METADATA_KEY)
    if metadata is None:
        return None
    metadata = _mapping(_METADATA_KEY, metadata)
    expected = {
        "version", "accumulation_steps", "precision",
        "gradient_clip_val", "gradient_clip_algorithm",
    }
    if set(metadata) != expected:
        raise ValueError("checkpoint resume metadata lacks required training settings")
    if metadata["version"] != _METADATA_VERSION:
        raise ValueError("unsupported resume metadata version")
    return _SavedResumeMetadata(
        _positive_integer("saved accumulation_steps", metadata["accumulation_steps"]),
        ResumeRuntime(
            metadata["precision"],
            metadata["gradient_clip_val"],
            metadata["gradient_clip_algorithm"],
        ),
    )


def _source_accumulation(
    metadata: Optional[_SavedResumeMetadata], explicit: Optional[int]
) -> int:
    recorded = None if metadata is None else metadata.accumulation_steps
    if explicit is not None:
        explicit = _positive_integer("source_accumulation", explicit)
        if recorded is not None and explicit != recorded:
            raise ValueError("source_accumulation differs from checkpoint metadata")
        return explicit
    if recorded is None:
        raise ValueError(
            "checkpoint has no accumulation metadata; supply source_accumulation explicitly"
        )
    return recorded


def _fit_state(checkpoint: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    loops = _mapping("loops", checkpoint.get("loops"))
    fit = _mapping("loops.fit_loop", loops.get("fit_loop"))
    state = _mapping("loops.fit_loop.state_dict", fit.get("state_dict"))
    loader_states = state.get("combined_loader")
    if not isinstance(loader_states, list) or len(loader_states) != 1:
        raise ValueError("resume requires exactly one stateful training dataloader")
    sampler = _mapping("combined_loader[0]", loader_states[0])
    if set(sampler) != _SAMPLER_FIELDS or sampler["version"] != 1:
        raise ValueError("unsupported training sampler checkpoint schema")
    return fit, sampler


def _epoch_batches(dataset_size: int, world_size: int, batch_size: int) -> int:
    batches = math.ceil(dataset_size / world_size) // batch_size
    if batches < 1:
        raise ValueError("training layout has no complete local batch")
    return batches


def _progress_group(name: str, value: Any) -> dict[str, int]:
    value = _mapping(name, value)
    if set(value) != _PROGRESS_FIELDS:
        raise ValueError(f"unsupported Lightning 2.5.5 progress fields: {name}")
    for field in _PROGRESS_FIELDS:
        _counter(f"{name}.{field}", value[field])
    return cast(dict[str, int], value)


def _translate_progress(group: dict[str, int], old_anchor: int, new_anchor: int) -> None:
    if (group["ready"] != old_anchor or group["started"] != old_anchor
            or group["processed"] != old_anchor
            or group["completed"] not in {old_anchor, old_anchor - 1}):
        raise ValueError("checkpoint was not saved at a completed training-batch phase")
    for field in _PROGRESS_FIELDS:
        translated = new_anchor + group[field] - old_anchor
        if translated < 0:
            raise ValueError("translated Lightning batch progress is negative")
        group[field] = translated


def _normalize_partial_epoch_progress(
    fit: Mapping[str, Any], epoch: int, completed_batches: int
) -> None:
    if completed_batches == 0:
        return
    progress = _mapping("epoch_progress", fit.get("epoch_progress"))
    for name in ("current", "total"):
        group = _progress_group(f"epoch_progress.{name}", progress.get(name))
        if group["ready"] != epoch + 1 or group["started"] != epoch + 1:
            raise ValueError("Lightning epoch progress differs from the sampler cursor")
        phase = (group["processed"], group["completed"])
        if phase == (epoch + 1, epoch + 1):
            # Trainer.save_checkpoint after fit(max_steps=...) sees on_train_end
            # close a partial epoch. Reopen it so Lightning restores mid-epoch.
            group["processed"] = group["completed"] = epoch
        elif phase != (epoch, epoch):
            raise ValueError("unsupported Lightning partial-epoch progress phase")


def _validate_full_checkpoint(checkpoint: Mapping[str, Any], fit: Mapping[str, Any]) -> int:
    for field in ("optimizer_states", "lr_schedulers", "global_step"):
        if field not in checkpoint:
            raise ValueError(f"resume requires a full checkpoint with {field}")
    global_step = _counter("global_step", checkpoint["global_step"])
    progress = _mapping(
        "epoch_loop.automatic_optimization.optim_progress",
        fit.get("epoch_loop.automatic_optimization.optim_progress"),
    )
    optimizer = _mapping("optim_progress.optimizer", progress.get("optimizer"))
    step = _mapping("optim_progress.optimizer.step", optimizer.get("step"))
    total = _mapping("optim_progress.optimizer.step.total", step.get("total"))
    if _counter("optimizer step completed", total.get("completed")) != global_step:
        raise ValueError("checkpoint optimizer progress differs from global_step")
    return global_step


def _translate_checkpoint_progress(
    checkpoint: dict[str, Any],
    layout: ResumeLayout,
    source_k: int,
) -> None:
    """Validate and translate only the sampler and Lightning progress fields."""

    fit, sampler = _fit_state(checkpoint)
    global_step = _validate_full_checkpoint(checkpoint, fit)

    dataset_size = _positive_integer("saved dataset_size", sampler["dataset_size"])
    source_b = _positive_integer("saved batch_size", sampler["batch_size"])
    source_w = _positive_integer("saved world_size", sampler["world_size"])
    epoch = _counter("saved epoch", sampler["epoch"])
    completed = _counter("saved completed_batches", sampler["completed_batches"])
    if source_b != layout.local_batch_size:
        raise ValueError("resume requires an unchanged local batch size")
    if source_w * source_k != layout.world_size * layout.accumulation_steps:
        raise ValueError("resume requires an unchanged world_size * accumulation_steps")

    old_batches = _epoch_batches(dataset_size, source_w, source_b)
    new_batches = _epoch_batches(dataset_size, layout.world_size, layout.local_batch_size)
    old_epoch_size = old_batches * source_w * source_b
    new_epoch_size = new_batches * layout.world_size * layout.local_batch_size
    if old_epoch_size != new_epoch_size:
        raise ValueError("resume layouts have different effective epoch lengths")
    if completed >= old_batches:
        raise ValueError("saved sampler cursor is not canonical")
    _normalize_partial_epoch_progress(fit, epoch, completed)

    old_microbatch = source_w * source_b
    new_microbatch = layout.global_microbatch_size
    offset = completed * old_microbatch
    if offset % layout.global_batch_size:
        raise ValueError("checkpoint is not at a complete optimizer-update boundary")
    if offset % new_microbatch:
        raise ValueError("saved global sample offset cannot form a target microbatch")
    new_completed = offset // new_microbatch
    if new_completed >= new_batches:
        raise ValueError("translated sampler cursor is not canonical")

    old_updates_per_epoch = math.ceil(old_batches / source_k)
    new_updates_per_epoch = math.ceil(new_batches / layout.accumulation_steps)
    old_updates = epoch * old_updates_per_epoch + completed // source_k
    new_updates = epoch * new_updates_per_epoch + new_completed // layout.accumulation_steps
    if old_updates != global_step or new_updates != global_step:
        raise ValueError("checkpoint global_step is inconsistent with its data cursor")

    progress = _mapping(
        "epoch_loop.batch_progress", fit.get("epoch_loop.batch_progress")
    )
    current = _progress_group("epoch_loop.batch_progress.current", progress.get("current"))
    total = _progress_group("epoch_loop.batch_progress.total", progress.get("total"))
    old_total_anchor = epoch * old_batches + completed
    new_total_anchor = epoch * new_batches + new_completed
    if total["processed"] != old_total_anchor:
        raise ValueError("Lightning total batch progress differs from the sampler cursor")

    if completed:
        old_current_anchor, new_current_anchor = completed, new_completed
    elif current["processed"] == 0:
        old_current_anchor = new_current_anchor = 0
    elif current["processed"] == old_batches:
        old_current_anchor, new_current_anchor = old_batches, new_batches
    else:
        raise ValueError("Lightning current batch progress differs from the sampler cursor")
    _translate_progress(current, old_current_anchor, new_current_anchor)
    _translate_progress(total, old_total_anchor, new_total_anchor)
    if completed == 0 and old_current_anchor == old_batches:
        # The sampler is in the next epoch, but Lightning retains the preceding
        # epoch's current counters. Its max_steps restart path adds them to the
        # fresh fetcher, so canonicalize the completed phase before restoring.
        for field in _PROGRESS_FIELDS:
            current[field] = 0
        total["completed"] = new_total_anchor
        if "is_last_batch" in progress:
            progress["is_last_batch"] = False

    sampler["world_size"] = layout.world_size
    sampler["completed_batches"] = new_completed


def adapt_checkpoint_layout(
    checkpoint: dict[str, Any],
    layout: ResumeLayout,
    *,
    source_accumulation: Optional[int] = None,
    runtime: Optional[ResumeRuntime] = None,
) -> dict[str, Any]:
    """Adapt one loaded Lightning 2.5.5 checkpoint in place.

    The local batch size and effective global optimizer batch stay fixed.  The
    epoch must retain the same effective shuffled sequence length under both
    layouts, and checkpoints must describe a completed optimizer update.
    """

    metadata = _saved_metadata(checkpoint)
    if runtime is not None:
        if not isinstance(runtime, ResumeRuntime):
            raise TypeError("runtime must be ResumeRuntime or None")
        if metadata is None:
            raise ValueError(
                "checkpoint has no runtime resume metadata; original precision and "
                "gradient clipping settings are required"
            )
        if metadata.runtime != runtime:
            raise ValueError(
                "resume requires unchanged precision and gradient clipping settings"
            )
    source_k = _source_accumulation(metadata, source_accumulation)
    _translate_checkpoint_progress(checkpoint, layout, source_k)
    if metadata is not None:
        checkpoint[_METADATA_KEY]["accumulation_steps"] = layout.accumulation_steps
    return checkpoint


_IDENTITY_MAP_LOCATION: Callable[[Storage, str], Storage] = lambda storage, location: storage


class LayoutResumeCheckpointIO(CheckpointIO):
    """Delegate checkpoint storage and adapt loaded resume state in memory."""

    def __init__(
        self,
        layout: ResumeLayout,
        *,
        runtime: ResumeRuntime,
        source_accumulation: Optional[int] = None,
        delegate: Optional[CheckpointIO] = None,
    ) -> None:
        self.layout = layout
        if not isinstance(runtime, ResumeRuntime):
            raise TypeError("runtime must be ResumeRuntime")
        self.runtime = runtime
        self.source_accumulation = source_accumulation
        self.delegate = TorchCheckpointIO() if delegate is None else delegate

    def save_checkpoint(
        self, checkpoint: dict[str, Any], path: Pathish, storage_options: Optional[Any] = None
    ) -> None:
        self.delegate.save_checkpoint(checkpoint, path, storage_options=storage_options)

    def load_checkpoint(
        self, path: Pathish, map_location: MAP_LOCATION = _IDENTITY_MAP_LOCATION
    ) -> dict[str, Any]:
        checkpoint = self.delegate.load_checkpoint(path, map_location=map_location)
        # The multiprocessing launcher uses CheckpointIO for a temporary raw
        # state_dict when returning worker weights to its parent process.
        if (isinstance(checkpoint, OrderedDict)
                and not {"loops", "global_step", "optimizer_states"} & checkpoint.keys()):
            return checkpoint
        return adapt_checkpoint_layout(
            checkpoint,
            self.layout,
            source_accumulation=self.source_accumulation,
            runtime=self.runtime,
        )

    def remove_checkpoint(self, path: Pathish) -> None:
        self.delegate.remove_checkpoint(path)

    def teardown(self) -> None:
        self.delegate.teardown()


__all__ = [
    "LayoutResumeCheckpointIO",
    "ResumeLayout",
    "ResumeMetadata",
    "ResumeRuntime",
    "adapt_checkpoint_layout",
]
