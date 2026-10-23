"""Optimizer-update scheduling, checkpoint retention and run artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence, Union, cast

from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loops.training_epoch_loop import _TrainingEpochLoop
from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from training._types import (
    Checkpoint,
    Pathish,
    PendingRetention,
    TrainingBatch,
)
from training.token import (
    TokenTrainingModule,
)

if TYPE_CHECKING:
    from lightning.pytorch.loops.fetchers import _DataFetcher

    from weight_adapters.import_mdlm import MDLMImportReport
    from weight_adapters.initialize_family import (
        FamilyInitializationReport,
        OfficialMDLMInitialization,
    )


InitializationReport = Union["FamilyInitializationReport", "OfficialMDLMInitialization", "MDLMImportReport"]


class _UpdateValidationLoop(_TrainingEpochLoop):
    """Schedule validation in optimizer updates, including short epoch tails."""

    def __init__(self, trainer: Trainer, every_updates: int | None) -> None:
        super().__init__(trainer, min_steps=trainer.min_steps, max_steps=trainer.max_steps)
        self.every_updates = every_updates

    def _should_check_val_fx(self, data_fetcher: _DataFetcher | None) -> bool:
        del data_fetcher
        return (self.trainer.enable_validation and self.every_updates is not None
                and not self._should_accumulate() and self.global_step > 0
                and self.global_step % self.every_updates == 0)


class _UpdateCheckpoint(ModelCheckpoint):
    """Save periodic checkpoints only after an actual optimizer update."""

    _PENDING_RETENTION = "pending_periodic_retention"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._pending_retention: PendingRetention | None = None

    def on_save_checkpoint(self, trainer: Trainer, pl_module: LightningModule,
                           checkpoint: Checkpoint) -> None:
        del pl_module
        if (self.save_top_k <= 0 or trainer.global_step <= 0
                or self._every_n_train_steps < 1
                or trainer.global_step % self._every_n_train_steps != 0
                or self._should_skip_saving_checkpoint(trainer)):
            return
        monitor_candidates = self._monitor_candidates(trainer)
        current = monitor_candidates["step"]
        if any(score.item() == current.item() for score in self.best_k_models.values()):
            return
        # Best is written first; let it describe the periodic file written next.
        del_filepath = (self.kth_best_model_path
                        if len(self.best_k_models) == self.save_top_k else None)
        filepath = self._get_metric_interpolated_filepath_name(
            monitor_candidates, trainer, del_filepath)
        state = checkpoint["callbacks"][self.state_key]
        state[self._PENDING_RETENTION] = {
            "path": filepath, "score": current.detach().cpu(),
        }

    def load_state_dict(self, state_dict: Checkpoint) -> None:
        self._pending_retention = state_dict.pop(self._PENDING_RETENTION, None)
        super().load_state_dict(state_dict)

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_train_start(trainer, pl_module)
        pending, self._pending_retention = self._pending_retention, None
        if pending is None or not self.file_exists(pending["path"], trainer):
            return
        filepath, current = pending["path"], pending["score"]
        if filepath in self.best_k_models:
            return
        del_filepath = None
        if len(self.best_k_models) == self.save_top_k:
            del_filepath = self.kth_best_model_path
            self.best_k_models.pop(del_filepath)
        self.current_score = current
        self.best_k_models[filepath] = current
        if len(self.best_k_models) == self.save_top_k:
            self.kth_best_model_path = min(
                self.best_k_models, key=cast(Callable[[str], Tensor], self.best_k_models.get))
            self.kth_value = self.best_k_models[self.kth_best_model_path]
        self.best_model_path = max(self.best_k_models, key=cast(Callable[[str], Tensor], self.best_k_models.get))
        self.best_model_score = self.best_k_models[self.best_model_path]
        if (del_filepath
                and self._should_remove_checkpoint(trainer, del_filepath, filepath)):
            self._remove_checkpoint(trainer, del_filepath)

    def _save_update(self, trainer: Trainer) -> None:
        if (self._should_skip_saving_checkpoint(trainer)
                or self._every_n_train_steps < 1
                or trainer.global_step % self._every_n_train_steps != 0):
            return
        monitor_candidates = self._monitor_candidates(trainer)
        if self.save_last:
            self.last_model_path = self.format_checkpoint_name(
                monitor_candidates, self.CHECKPOINT_NAME_LAST)
        self._save_topk_checkpoint(trainer, monitor_candidates)
        self._save_last_checkpoint(trainer, monitor_candidates)

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                           outputs: STEP_OUTPUT, batch: TrainingBatch, batch_idx: int) -> None:
        if trainer.fit_loop._should_accumulate():
            return
        if cast(_UpdateValidationLoop, trainer.fit_loop.epoch_loop)._should_check_val_fx(None):
            return
        self._save_update(trainer)

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Save after validation and best-model selection at this update.
        self._save_update(trainer)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # _RunOutput owns the final full-state checkpoint.
        pass


class _RunOutput(Callback):
    def __init__(self, config: DictConfig, tokenizer: PreTrainedTokenizerBase, *,
                 final_path: Pathish, initialization: InitializationReport | None = None) -> None:
        self.config, self.tokenizer = config, tokenizer
        self.initialization = initialization
        self.final_path = final_path

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        root = Path(self.config.output_dir)
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        OmegaConf.save(self.config, root / ("config-" + stamp + ".yaml"), resolve=True)
        if not (root / "tokenizer").exists():
            self.tokenizer.save_pretrained(root / "tokenizer")
        if self.initialization is not None:
            report = {"source": str(self.config.startup.path),
                      "weights": self.config.startup.weights,
                      **_initialization_summary(self.initialization)}
            (root / "initialization.json").write_text(json.dumps(report, indent=2) + "\n")
        batch = self.config.runtime.local_batch_size
        accumulation = trainer.accumulate_grad_batches
        print("Training %s: local_batch=%d, world_size=%d, accumulation=%d, global_batch=%d, step=%d"
              % (cast(TokenTrainingModule[Any], pl_module).training_name, batch, trainer.world_size, accumulation,
                 batch * trainer.world_size * accumulation, trainer.global_step))

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        trainer.save_checkpoint(str(self.final_path), weights_only=False)


def _initialization_summary(report: InitializationReport) -> dict[str, Any]:
    """Keep import counts and newly initialized modules, not vocabulary-row lists."""
    result: dict[str, Any]
    if hasattr(report, "import_report"):
        official = cast("OfficialMDLMInitialization", report)
        result = {"import": _initialization_summary(official.import_report)}
        if official.family_report is not None:
            result["family"] = _initialization_summary(official.family_report)
        return result
    result = {}
    for name in ("source_family", "target_family", "seed", "source_entry_count"):
        if hasattr(report, name):
            result[name] = getattr(report, name)
    for name in ("copied_keys", "consumed_source_keys", "vocabulary_rows", "discarded_source_rows",
                 "initialized_vocabulary_rows", "initialized_target_rows"):
        if hasattr(report, name):
            result[name + "_count"] = len(getattr(report, name))
    if hasattr(report, "initialized_keys"):
        result["initialized_parameters"] = list(cast(Sequence[str], report.initialized_keys))
    return result
