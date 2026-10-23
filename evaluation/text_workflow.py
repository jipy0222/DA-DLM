"""Packed-text generation and independently repeatable corpus scoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence, cast

from omegaconf import DictConfig, OmegaConf

from data.text import make_text_generation_loader, prepare_text_generation
from data.types import Metadata
from training.configuration import require_path as _require_path

from .generation import make_text_options
from .runner import generate_dataset
from .text_metrics import compute_text_metrics
from .workflow import (
    GENERATORS,
    check_single_process,
    evaluation_settings,
    generation_settings,
    generation_statistics,
    load_generation_source,
    output_directory,
    publish_metrics,
    publish_results,
)


def _score(target: Path, records: Sequence[MutableMapping[str, Any]], generation: Mapping[str, Any], scoring: Mapping[str, Any]) -> Metadata:
    scores = compute_text_metrics(records, scoring)
    metrics = {**generation, **scores}
    publish_metrics(target, metrics)
    print(json.dumps(metrics, indent=2))
    return metrics


def evaluate_text(config: DictConfig) -> Metadata:
    """Save generation before scoring so failed metrics can be rerun independently."""
    check_single_process()
    _require_path("data.corpus_path", config.data.corpus_path)
    target = output_directory(config.output_dir)
    evaluation = config.evaluation
    if evaluation.split != "validation":
        raise ValueError("text evaluation uses the corpus profile's validation split")
    source, tokenizer = load_generation_source(config)
    prompt_length = evaluation.prompt_length
    if type(prompt_length) is not int or prompt_length < 0:
        raise ValueError("evaluation.prompt_length must be a nonnegative integer")
    if prompt_length == 0 and not config.data.window_bos_eos:
        raise ValueError("unprompted text evaluation requires window_bos_eos to count the given BOS")
    options = make_text_options(
        source.tokens, has_prompt=prompt_length > 0,
        **generation_settings(config, source, default_length=None,
                              stop_token_id=source.tokens.eos_token_id))
    length = options["generation_length"]
    if type(length) is not int or length < 1:
        raise ValueError("generation.generation_length must be a positive integer")
    total = prompt_length + length
    capacity = min(source.model._content_limit,
                   source.model.config.backbone.max_attention_length)
    if total > capacity:
        raise ValueError(f"prompt plus generation length {total} exceeds checkpoint capacity {capacity}")
    store_dir = Path(evaluation.store_dir or target.with_name(target.name + "_data")).expanduser().resolve()
    if target == store_dir or target in store_dir.parents or store_dir in target.parents:
        raise ValueError("evaluation.store_dir and output_dir must be separate directories")
    prepared = prepare_text_generation(
        config.data.corpus_path, tokenizer, source.tokens,
        profile=config.data.profile, revision=config.data.revision,
        tokenizer_id=config.data.tokenizer_id, store_dir=store_dir,
        prompt_length=prompt_length, generation_length=length,
        num_samples=evaluation.num_samples, shuffle=evaluation.shuffle, seed=config.seed,
        pack_batch_size=config.data.packing_batch_size,
        num_workers=evaluation.preparation_num_workers, column=config.data.column,
        document_eos=config.data.document_eos, window_bos_eos=config.data.window_bos_eos)
    loader = make_text_generation_loader(
        prepared, eval_batch_size=evaluation.batch_size, num_workers=evaluation.num_workers,
        seed=config.seed, pin_memory=evaluation.pin_memory)
    result = generate_dataset(
        loader, GENERATORS[source.family], source.model, tokenizer,
        generation_kwargs=options, device=evaluation.device, precision=evaluation.precision,
        seed=config.seed, pad_token_id=source.tokens.pad_token_id, include_prompt=True)
    generation = {
        "family": source.family, "weights": source.weights_kind,
        "source_count": prepared.source_count, "eligible_count": prepared.eligible_count,
        "prompt_length": prompt_length, "generation_length": length,
        "total_length": total,
        **generation_statistics(result, fixed_start=options["start_token_id"] is not None),
    }
    settings = evaluation_settings(config, source, options)
    settings["effective"]["store_dir"] = str(Path(store_dir).resolve())
    settings["effective"]["generation_seed"] = config.seed
    publish_results(target, result.records, settings, generation=generation)
    print(f"Generated records saved to {target}; computing requested metrics.")
    records = result.records
    del source, tokenizer, loader, prepared, result
    return _score(target, records, generation,
                  cast(dict[str, Any], OmegaConf.to_container(evaluation.scoring, resolve=True)))


def score_text(config: DictConfig) -> Metadata:
    """Score saved text records in a new directory, without a generator or raw corpus."""
    check_single_process()
    _require_path("evaluation.records_path", config.evaluation.records_path)
    target = output_directory(config.output_dir)
    source = Path(config.evaluation.records_path).expanduser().resolve(strict=True)
    with (source / "predictions.jsonl").open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    generation = json.loads((source / "generation.json").read_text(encoding="utf-8"))
    settings = cast(dict[str, Any], OmegaConf.to_container(OmegaConf.load(source / "config.yaml"), resolve=True))
    scoring = cast(dict[str, Any], OmegaConf.to_container(config.evaluation.scoring, resolve=True))
    settings.update(mode="score_text", seed=config.seed, output_dir=str(target), scored_from=str(source))
    settings["evaluation"]["scoring"] = scoring
    settings["evaluation"]["records_path"] = str(source)
    publish_results(target, records, settings, generation=generation)
    return _score(target, records, generation, scoring)
