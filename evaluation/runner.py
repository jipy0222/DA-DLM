"""Dataset-level execution around a caller-selected generation function."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Literal, Sequence, cast

import torch

from generation.types import GenerationResult
from progress import track

from .generation import generate_batch_records

Precision = Literal["fp32", "bf16"]
_COUNTER_FIELDS = (
    "prediction_nfe", "prefill_nfe", "commit_nfe", "total_nfe", "update_steps",
)


@dataclass(frozen=True)
class GenerationTotals:
    """Summed per-sample work; batch fields exist only for raw execution."""

    sample_count: int
    prediction_nfe: int
    prefill_nfe: int
    commit_nfe: int
    total_nfe: int
    update_steps: int
    batch_count: int | None
    batch_calls: dict[str, int] | None


@dataclass(frozen=True)
class DatasetGenerationResult:
    records: list[dict[str, Any]]
    raw_stats: GenerationTotals
    scored_stats: GenerationTotals


def _model_device(model: Any, requested: torch.device | str | None) -> torch.device:
    if requested is not None:
        model.to(torch.device(requested))
    return cast('torch.device', next(model.parameters()).device)


def _generation_generator(
    generation_kwargs: Mapping[str, Any],
    generator: torch.Generator | None,
    seed: int | None,
    device: torch.device,
) -> dict[str, Any]:
    if generator is not None and seed is not None:
        raise ValueError("generator and seed are mutually exclusive")
    if "generator" in generation_kwargs and (generator is not None or seed is not None):
        raise ValueError("supply generator in one place only")
    kwargs = dict(generation_kwargs)
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        generator = torch.Generator(device=device).manual_seed(seed)
    if generator is not None:
        kwargs["generator"] = generator
    selected = kwargs.get("generator")
    if selected is not None:
        if not isinstance(selected, torch.Generator):
            raise TypeError("generator must be a torch.Generator")
        generator_device = torch.device(selected.device)
        if generator_device.type == device.type and generator_device.index is None:
            generator_device = device
        if generator_device != device:
            raise ValueError("generator and model must share a device")
    return kwargs


def _counter_rows(result: GenerationResult) -> dict[str, list[int]]:
    stats = result.stats
    values = torch.stack((
        stats.prediction_nfe, stats.prefill_nfe, stats.commit_nfe,
        stats.total_nfe, stats.update_steps,
    )).detach().cpu().tolist()
    return {
        name: [int(item) for item in row]
        for name, row in zip(_COUNTER_FIELDS, values)
    }


def _sample_ids(batch: Mapping[str, Any], batch_size: int) -> Sequence[Any] | torch.Tensor | None:
    values = batch.get("sample_id")
    if values is None:
        return None
    if isinstance(values, torch.Tensor) and values.ndim == 0:
        values = values.reshape(1)
    if len(values) != batch_size:
        raise ValueError("sample_id batch size disagrees with prompt batch")
    return values


def _sample_id(values: Sequence[Any] | torch.Tensor, row: int) -> Any:
    value = values[row]
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("each sample_id tensor must be scalar")
        return value.detach().cpu().item()
    return value


def _totals(
    sums: Mapping[str, int], sample_count: int, *,
    batch_count: int | None, batch_calls: dict[str, int] | None,
) -> GenerationTotals:
    return GenerationTotals(
        sample_count=sample_count,
        prediction_nfe=sums["prediction_nfe"],
        prefill_nfe=sums["prefill_nfe"],
        commit_nfe=sums["commit_nfe"],
        total_nfe=sums["total_nfe"],
        update_steps=sums["update_steps"],
        batch_count=batch_count,
        batch_calls=batch_calls,
    )


def generate_dataset(
    loader: Iterable[Mapping[str, Any]],
    generate: Callable[..., GenerationResult],
    model: Any,
    tokenizer: Any,
    *,
    generation_kwargs: Mapping[str, Any],
    device: torch.device | str | None = None,
    precision: Precision = "fp32",
    generator: torch.Generator | None = None,
    seed: int | None = None,
    ema: Any = None,
    include_prompt: bool = False,
    pad_token_id: int | None = None,
) -> DatasetGenerationResult:
    """Generate every supplied batch without gathering, checkpoint I/O, or writes.

    An explicit device moves the model once; precision controls autocast without
    converting stored weights. The loader remains caller-owned.
    Raw totals include distributed padding copies and actual batch calls. Scored
    totals include only returned, nonzero-weight rows; filtered batch calls have
    no well-defined value and are therefore ``None``.
    """
    if precision not in ("fp32", "bf16"):
        raise ValueError("precision must be 'fp32' or 'bf16'")
    model_device = _model_device(model, device)
    kwargs = _generation_generator(
        generation_kwargs, generator, seed, model_device,
    )
    raw_sums = {name: 0 for name in _COUNTER_FIELDS}
    scored_sums = raw_sums.copy()
    batch_calls: dict[str, int] = {}
    records: list[dict[str, Any]] = []
    raw_count = 0
    batch_count = 0
    ema_context = ema.average_parameters(model) if ema is not None else nullcontext()

    with ema_context, torch.autocast(
        model_device.type, dtype=torch.bfloat16, enabled=precision == "bf16",
    ):
        for batch_index, original_batch in enumerate(
                track(loader, desc="Generating", unit="batch", leave=True)):
            batch = dict(original_batch)
            prompt_ids = batch.get("prompt_ids")
            if not isinstance(prompt_ids, torch.Tensor):
                raise TypeError("batch prompt_ids must be a tensor")
            batch["prompt_ids"] = prompt_ids.to(model_device)
            ids = _sample_ids(batch, prompt_ids.shape[0])
            raw, batch_records = generate_batch_records(
                generate, model, tokenizer, batch,
                generation_kwargs=kwargs, ema=None,
                include_prompt=include_prompt, pad_token_id=pad_token_id,
            )
            counters = _counter_rows(raw)
            size = raw.token_ids.shape[0]
            raw_count += size
            batch_count += 1
            for name in _COUNTER_FIELDS:
                raw_sums[name] += sum(counters[name])
            for phase, count in raw.stats.batch_calls.items():
                batch_calls[phase] = batch_calls.get(phase, 0) + int(count)

            for record in batch_records:
                row = record["row_index"]
                record["batch_index"] = batch_index
                for name in _COUNTER_FIELDS:
                    value = counters[name][row]
                    record[name] = value
                    scored_sums[name] += value
                if ids is not None:
                    record["sample_id"] = _sample_id(ids, row)
                records.append(record)

    if batch_count == 0:
        raise ValueError("generation loader produced no batches")
    return DatasetGenerationResult(
        records=records,
        raw_stats=_totals(
            raw_sums, raw_count, batch_count=batch_count, batch_calls=batch_calls,
        ),
        scored_stats=_totals(
            scored_sums, len(records), batch_count=None, batch_calls=None,
        ),
    )


__all__ = [
    "DatasetGenerationResult", "GenerationTotals", "Precision", "generate_dataset",
]
