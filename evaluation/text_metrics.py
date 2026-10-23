"""Optional open-text metrics over saved generation records."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from typing import Any, MutableMapping, Sequence, cast

from progress import progress_bar

from .language_model import (
    featurize_texts,
    load_causal_lm,
    load_tokenizer,
    prepare_causal_inputs,
    score_causal_inputs,
    tokenize_texts,
)

_METRICS = frozenset({"genppl", "mauve", "coherence", "entropy", "diversity"})


def _positive_int(config: Mapping[str, Any], name: str, default: int) -> int:
    value = config.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _model_path(config: Mapping[str, Any]) -> str:
    value = config.get("model_path")
    if value is None or not str(value):
        raise ValueError("model_path is required for the selected text metrics")
    return str(value)


def _texts(records: Sequence[Mapping[str, Any]], key: str) -> list[str]:
    values = []
    for index, record in enumerate(records):
        if key not in record:
            raise KeyError(f"record {index} has no {key}")
        value = record[key]
        if not isinstance(value, str):
            raise TypeError(f"record {index} {key} must be a string")
        values.append(value)
    return values


def _continuation_starts(records: Sequence[Mapping[str, Any]]) -> list[int]:
    starts = []
    for index, record in enumerate(records):
        if "continuation_start" not in record:
            prompt_ids = record.get("prompt_ids", ())
            prompt_text = record.get("prompt_text", "")
            if len(prompt_ids) == 0 and prompt_text == "":
                starts.append(0)
                continue
            raise ValueError(
                f"prompted record {index} requires an explicit continuation_start"
            )
        starts.append(record["continuation_start"])
    return cast('list[int]', starts)


def _entropy(token_ids: Sequence[int]) -> float:
    if not token_ids:
        return 0.0
    counts = Counter(token_ids)
    total = len(token_ids)
    return -sum((count / total) * math.log(count / total) for count in counts.values())


def _diversity(token_ids: Sequence[int]) -> float:
    result = 1.0
    for n in range(2, 5):
        if len(token_ids) < n:
            ratio = 1.0
        else:
            ngrams = [tuple(token_ids[index:index + n])
                      for index in range(len(token_ids) - n + 1)]
            ratio = len(set(ngrams)) / len(ngrams)
        result *= ratio
    return 100.0 * result


def _mauve_parameters(
    sample_count: int, config: Mapping[str, Any],
) -> tuple[int, int, int]:
    requested = config.get("mauve_num_buckets")
    if requested is None:
        buckets = sample_count // 20
        if buckets < 1:
            raise ValueError("default MAUVE bucketing requires at least 20 samples")
    else:
        if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
            raise ValueError("mauve_num_buckets must be a positive integer or null")
        buckets = requested
    max_length = _positive_int(config, "mauve_max_length", 1024)
    seed = config.get("seed", 25)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return buckets, max_length, seed


def _compute_mauve(
    records: Sequence[Mapping[str, Any]], generated: Sequence[str],
    batch_size: int, device: str,
    model: Any, tokenizer: Any, parameters: tuple[int, int, int],
) -> tuple[float, float]:
    references = _texts(records, "reference_full_text")
    buckets, max_length, seed = parameters

    reference_features = featurize_texts(
        model, tokenizer, references, batch_size=batch_size,
        max_length=max_length, device=device,
    ).numpy()
    generated_features = featurize_texts(
        model, tokenizer, generated, batch_size=batch_size,
        max_length=max_length, device=device,
    ).numpy()

    import mauve

    with progress_bar(desc="Computing MAUVE", total=1, unit="metric") as bar:
        output = mauve.compute_mauve(
            p_features=reference_features,
            q_features=generated_features,
            num_buckets=buckets,
            verbose=False,
            seed=seed,
        )
        if bar is not None:
            bar.update()
    return float(output.mauve), float(output.frontier_integral)


def compute_text_metrics(
    records: Sequence[MutableMapping[str, Any]], config: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute selected metrics and attach available per-sample values."""
    enabled_value = config.get("enabled", ())
    if isinstance(enabled_value, (str, bytes)):
        raise TypeError("enabled must be a sequence of metric names")
    enabled = set(enabled_value)
    unknown = enabled - _METRICS
    if unknown:
        raise ValueError(f"unknown text metrics: {sorted(unknown)}")

    result: dict[str, Any] = {"sample_count": len(records)}
    if not enabled:
        return result
    if not records:
        raise ValueError("enabled text metrics require at least one record")

    mauve_parameters = (_mauve_parameters(len(records), config)
                        if "mauve" in enabled else None)

    batch_size = _positive_int(config, "batch_size", 8)
    stride = _positive_int(config, "stride", 512)
    max_length = config.get("max_length")
    if max_length is not None and (
        isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1
    ):
        raise ValueError("max_length must be a positive integer or null")
    device = str(config.get("device", "cpu"))
    generated = _texts(records, "full_text")

    tokenizer_path = config.get("tokenizer_path")
    if tokenizer_path is None:
        tokenizer_path = _model_path(config)
    elif not str(tokenizer_path):
        raise ValueError("tokenizer_path must not be empty")
    tokenizer = load_tokenizer(str(tokenizer_path))

    model = None
    if enabled & {"genppl", "coherence", "mauve"}:
        model = load_causal_lm(_model_path(config), device=device)

    inputs = None
    if enabled & {"genppl", "coherence", "entropy"}:
        starts = _continuation_starts(records) if "coherence" in enabled else None
        inputs = prepare_causal_inputs(
            tokenizer, generated, continuation_starts=starts, max_length=max_length,
        )
    if enabled & {"genppl", "coherence"}:
        assert inputs is not None
        scores = score_causal_inputs(
            model, tokenizer, inputs,
            batch_size=batch_size,
            stride=stride,
            device=device,
        )
        if "genppl" in enabled:
            token_count = sum(scores.full_token_counts)
            if token_count == 0:
                raise ValueError("genPPL requires at least one scored token")
            total_nll = sum(scores.full_nll_sums)
            mean_nll = total_nll / token_count
            result.update(
                gen_nll=mean_nll,
                genppl=math.exp(mean_nll),
                genppl_token_count=token_count,
            )
            for record, nll_sum, count in zip(
                records, scores.full_nll_sums, scores.full_token_counts,
            ):
                if count == 0:
                    raise ValueError("each genPPL record must contain a scored token")
                average = nll_sum / count
                record.update(
                    gen_nll=average, genppl=math.exp(average),
                    genppl_token_count=count,
                )

        if "coherence" in enabled:
            continuation_nll_sums = cast(tuple[float, ...], scores.continuation_nll_sums)
            continuation_token_counts = cast(tuple[int, ...], scores.continuation_token_counts)
            values = []
            for index, (record, nll_sum, count) in enumerate(zip(
                records, continuation_nll_sums, continuation_token_counts,
            )):
                if count == 0:
                    raise ValueError(
                        f"coherence record {index} has no continuation token after tokenization"
                    )
                value = nll_sum / count
                values.append(value)
                record.update(coherence=value, coherence_token_count=count)
            result.update(
                coherence=sum(values) / len(values),
                coherence_sample_count=len(values),
                coherence_token_count=sum(continuation_token_counts),
            )

    if "entropy" in enabled:
        assert inputs is not None
        rows = inputs.token_rows
        # These rows are not padded; every emitted ID, including a real EOS that
        # may share its ID with PAD, is part of the sample's token distribution.
        values = [_entropy(row) for row in rows]
        for record, value in zip(records, values):
            record["entropy"] = value
        result["entropy"] = sum(values) / len(values)

    if "diversity" in enabled:
        # Slice at the recorded joint-decode boundary, not an independently
        # decoded target that may start inside a multi-byte character.
        continuations = [text[start:] for text, start in zip(
            generated, _continuation_starts(records))]
        rows = tokenize_texts(tokenizer, continuations, add_special_tokens=False)
        values = [_diversity(row) for row in rows]
        for record, value in zip(records, values):
            record["diversity"] = value
        result["diversity"] = sum(values) / len(values)

    if "mauve" in enabled:
        score, frontier = _compute_mauve(
            records, generated, batch_size, device, model, tokenizer,
            cast(tuple[int, int, int], mauve_parameters),
        )
        result.update(mauve=score, frontier_integral=frontier)

    return result


__all__ = ["compute_text_metrics"]
