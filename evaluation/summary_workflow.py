"""Single-process summary generation and ROUGE evaluation from native checkpoints."""

from __future__ import annotations

import json

from omegaconf import DictConfig

from data.summary import (
    SummaryConfig,
    load_summary_split,
    make_summary_generation_loader,
    prepare_summary_generation,
)
from data.types import Metadata
from training.configuration import require_path as _require_path

from .generation import make_summary_options
from .rouge import compute_record_rouge
from .runner import generate_dataset
from .workflow import (
    GENERATORS,
    check_single_process,
    evaluation_settings,
    generation_settings,
    generation_statistics,
    load_generation_source,
    output_directory,
    publish_results,
)


def evaluate_summary(config: DictConfig) -> Metadata:
    """Evaluate one checkpoint/seed and publish predictions, metrics and effective settings."""
    check_single_process()
    _require_path("data.corpus_path", config.data.corpus_path)
    limit = config.evaluation.max_examples
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("evaluation.max_examples must be a positive integer or null")
    target = output_directory(config.output_dir)
    evaluation = config.evaluation
    source, tokenizer = load_generation_source(config)
    if source.task is None or source.task["kind"] != "summary":
        raise ValueError("summary evaluation requires a summary checkpoint with task metadata")
    for field in ("profile", "revision"):
        if config.data[field] != source.task[field]:
            raise ValueError(f"evaluation data.{field} differs from the summary checkpoint")
    summary = SummaryConfig(**source.task["summary"])
    options = make_summary_options(source.tokens, **generation_settings(
        config, source, default_length=summary.max_target_tokens,
        stop_token_id=source.tokens.summary_eos))
    raw = load_summary_split(config.data.corpus_path, split=evaluation.split)
    prepared = prepare_summary_generation(
        raw, tokenizer, source.tokens, summary,
        num_workers=evaluation.preparation_num_workers, include_sample_id=True)
    eligible_count = len(prepared)
    if not eligible_count:
        raise ValueError("summary evaluation has no valid examples")
    if evaluation.shuffle:
        prepared = prepared.shuffle(seed=config.seed)
    if limit is not None:
        prepared = prepared.select(range(min(limit, eligible_count)))
    loader = make_summary_generation_loader(
        prepared, eval_batch_size=evaluation.batch_size, num_workers=evaluation.num_workers,
        seed=config.seed, pin_memory=evaluation.pin_memory)
    result = generate_dataset(
        loader, GENERATORS[source.family], source.model, tokenizer,
        generation_kwargs=options, device=evaluation.device, precision=evaluation.precision,
        seed=config.seed, pad_token_id=source.tokens.pad_token_id,
        include_prompt=config.generation.get("include_prompt", False))
    rouge = compute_record_rouge(result.records)
    for record, scores in zip(result.records, rouge.per_sample):
        record.update(scores)
    metrics = {
        "family": source.family, "weights": source.weights_kind,
        "source_count": len(raw), "eligible_count": eligible_count, **rouge.macro,
        **generation_statistics(result, fixed_start=True),
    }
    publish_results(target, result.records, evaluation_settings(config, source, options), metrics=metrics)
    print(json.dumps(metrics, indent=2))
    return metrics
