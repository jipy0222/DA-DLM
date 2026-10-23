"""Shared checkpoint, generation settings, statistics and result publication."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence, cast

from omegaconf import DictConfig, OmegaConf

from data.tokenizer_io import check_vocabulary as _check_vocabulary
from data.tokenizer_io import load_tokenizer as _tokenizer
from data.tokens import prepare_tokenizer
from data.types import Metadata, Pathish
from generation import generate_ar, generate_bd3lm, generate_dadlm, generate_mdlm
from generation.config import (
    DADLMGenerationConfig,
    DiffusionConfig,
    SamplingConfig,
    StopConfig,
)
from training.checkpoint import checkpoint_tokenizer as _checkpoint_tokenizer
from training.checkpoint import load_native_checkpoint

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from evaluation.runner import DatasetGenerationResult
    from models import BD3LMConfig, DADLMConfig
    from training.checkpoint import NativeCheckpointLoad


GENERATORS = {"ar": generate_ar, "mdlm": generate_mdlm,
              "bd3lm": generate_bd3lm, "dadlm": generate_dadlm}


def check_single_process() -> None:
    tasks = os.environ.get("SLURM_STEP_NUM_TASKS", os.environ.get("SLURM_NTASKS", "1"))
    if (int(os.environ.get("WORLD_SIZE", "1")) != 1
            or ("SLURM_PROCID" in os.environ and int(tasks) != 1)):
        raise ValueError("generation evaluation requires a single process; do not launch multiple ranks")


def output_directory(value: Pathish) -> Path:
    if not value:
        raise ValueError("output_dir is required")
    target = Path(value).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"evaluation output already exists: {target}")
    return target


def load_generation_source(config: DictConfig) -> tuple[NativeCheckpointLoad, PreTrainedTokenizerBase]:
    """Load selected native weights and their companion tokenizer, without an optimizer."""
    if config.startup.source_kind != "native":
        raise ValueError("generation evaluation requires a native checkpoint")
    if not config.startup.path:
        raise ValueError("startup.path is required")
    source = load_native_checkpoint(config.startup.path, weights=config.evaluation.weights)
    tokenizer = _checkpoint_tokenizer(config.startup)
    if prepare_tokenizer(tokenizer) != source.tokens:
        raise ValueError("evaluation tokenizer differs from checkpoint token roles")
    companion = source.checkpoint_path.parent.parent / "tokenizer"
    if config.startup.source_tokenizer_path and companion.is_dir():
        _check_vocabulary(tokenizer, source.tokens, _tokenizer(companion))
    return source, tokenizer


def generation_settings(config: DictConfig, source: NativeCheckpointLoad, *, default_length: int | None, stop_token_id: int) -> dict[str, Any]:
    """Resolve family-specific typed settings; task wrappers supply start and stop roles."""
    values = cast(dict[str, Any], OmegaConf.to_container(config.generation, resolve=True))
    length = values["generation_length"]
    if length is None:
        length = default_length
    sampling = values["sampling"]
    sampling["blocked_token_ids"] = tuple(sampling.get("blocked_token_ids", ()))
    stop = values["stop"]
    stop["token_ids"] = tuple(stop.get(
        "token_ids", () if stop["mode"] == "none" else (stop_token_id,)))
    diffusion = values.get("diffusion")
    dag = values.get("dag")
    if source.family == "ar":
        if diffusion is not None:
            raise ValueError("AR generation does not accept diffusion settings")
    else:
        if diffusion is None:
            raise ValueError(f"{source.family} generation requires diffusion settings")
        if diffusion["num_steps"] is None:
            diffusion["num_steps"] = (length if source.family == "mdlm"
                                      else cast("BD3LMConfig | DADLMConfig", source.model.config).block_size)
        diffusion = DiffusionConfig(**diffusion)
    if source.family == "dadlm":
        if dag is None:
            raise ValueError("DADLM generation requires DAG settings")
        dag = DADLMGenerationConfig(**dag)
    elif dag is not None:
        raise ValueError("DAG generation settings require a DADLM checkpoint")
    return dict(generation_length=length, sampling=SamplingConfig(**sampling),
                stop=StopConfig(**stop), diffusion=diffusion, dag=dag)


def generation_statistics(result: DatasetGenerationResult, *, fixed_start: bool) -> Metadata:
    """Report actual per-sample work and predicted-token yield, excluding a given start."""
    count = len(result.records)
    for record in result.records:
        nfe = record["prediction_nfe"]
        record["tpf"] = (record["length"] - int(fixed_start)) / nfe if nfe else 0.0
    mean_length = sum(row["length"] for row in result.records) / count
    return {
        "sample_count": count,
        "tpf": sum(row["tpf"] for row in result.records) / count,
        "mean_length": mean_length, "mean_new_tokens": mean_length - int(fixed_start),
        "mean_prediction_nfe": result.scored_stats.prediction_nfe / count,
        "mean_total_nfe": result.scored_stats.total_nfe / count,
        "raw_stats": asdict(result.raw_stats), "scored_stats": asdict(result.scored_stats),
    }


def evaluation_settings(config: DictConfig, source: NativeCheckpointLoad, options: Mapping[str, Any]) -> Metadata:
    fields = ("mode", "seed", "output_dir", "startup", "data", "evaluation", "generation")
    saved = {name: (OmegaConf.to_container(config[name], resolve=True)
                    if OmegaConf.is_config(config[name]) else config[name]) for name in fields}
    saved["effective"] = {
        "checkpoint": str(source.checkpoint_path),
        "family": source.family, "weights": source.weights_kind,
        "model": source.training_contract["model"], "task": source.task,
        "tokens": source.tokens.to_dict(),
        "generation": {key: asdict(value) if is_dataclass(value) else value
                       for key, value in options.items()},
    }
    return saved


def publish_results(target: Path, records: Sequence[Mapping[str, Any]], settings: Mapping[str, Any], *, metrics: Mapping[str, Any] | None=None, generation: Mapping[str, Any] | None=None) -> None:
    """Publish one new result directory; an existing result is never overwritten."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{target.name}.writing-", dir=target.parent) as directory:
        temporary = Path(directory)
        with (temporary / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        for name, value in (("metrics", metrics), ("generation", generation)):
            if value is not None:
                (temporary / f"{name}.json").write_text(
                    json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        OmegaConf.save(OmegaConf.create(cast(dict[str, Any], settings)), temporary / "config.yaml")
        if target.exists():
            raise FileExistsError(f"evaluation output already exists: {target}")
        os.rename(temporary, target)


def publish_metrics(target: Path, metrics: Mapping[str, Any]) -> None:
    """Add scores to a published generation run after scoring completes."""
    with tempfile.TemporaryDirectory(prefix=".scoring-", dir=target) as directory:
        temporary = Path(directory) / "metrics.json"
        temporary.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.link(temporary, target / "metrics.json")
